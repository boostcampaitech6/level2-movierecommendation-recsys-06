from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp


#이미 완성된 MultiDAE의 코드를 참고하여 그 아래 MultiVAE의 코드를 완성해보세요!
class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input):
        # h = F.normalize(input)
        h = self.drop(input)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)



class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1] # default: 대칭 구조

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        # h = F.normalize(input)
        h = self.drop(input)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else: #
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar): # latent factor(모수) 조정
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) # std와 동일 모양의 N(0,1)난수생성
            return eps.mul(std).add_(mu) # train: 난수*std + mu
        else:
            return mu # val/test: mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0) # dropout한 old encoder의 latent factor
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior) # original: N(0,1)=N(0,exp(1))
        post_prior = log_norm_pdf(z, post_mu, post_logvar) # Z <-> Z_old
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior) # N(0,exp(10))
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)] # mixing (log가중치)
        
        density_per_gaussian = torch.stack(gaussians, dim=-1) 
        
        return torch.logsumexp(density_per_gaussian, dim=-1) # log(sum(exp(x)))


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        # dense connection
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))

        # residual connection
        # h1 = self.ln1(swish(self.fc1(x)))
        # h2 = self.ln2(swish(self.fc2(h1) + h1))
        # h3 = self.ln3(swish(self.fc3(h2) + h2))
        # h4 = self.ln4(swish(self.fc4(h3) + h3))
        # h5 = self.ln5(swish(self.fc5(h4) + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    

class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(RecVAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)    
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        
        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            
            return x_pred, negative_elbo
            
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


# 1. encoder 3epoch 학습
#   전체 forward
#   old encoder load
#   composite prior loss: KL-diverence
#   update encoder
  
# 2. decoder 1epoch 학습
#   최신 encoder -> decoder
#   reconstruction loss
#   update decoder
        
        

def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1)) # softmax: multinomial의 loss
    # BCE = torch.nn.BCEWithLogitsLoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) # 

    return BCE + anneal * KLD

def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # BCE = torch.nn.BCEWithLogitsLoss(reduction='sum')(recon_x, x)
    return BCE


# closed form: fitting -> predict from input
class EASE(nn.Module):
    def __init__(self, n_users, n_items):
        super(EASE, self).__init__()

        self.iid_name = 'item'
        self.uid_name = 'user'

        self.user_num = n_users
        self.item_num = n_items

        self.reg_weight = 500 # 이거 계속 바꿔봐야 해!

        self.topk = 10

    def fit(self, X):
        '''
        fit closed form parameters
        X: sparse matrix (user_num * item_num)
        '''

        G = X.T @ X # item_num * item_num
        G += self.reg_weight * sp.identity(G.shape[0])
        G = G.todense() # why not just use scipy?

        P = np.linalg.inv(G)
        B = -P / np.diag(P) # equation 8 in paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
        np.fill_diagonal(B, 0.)

        self.item_similarity = B # item_num * item_num
        self.item_similarity = np.array(self.item_similarity)
        self.interaction_matrix = X # user_num * item_num

    def predict(self, u, i):
        return self.interaction_matrix[u, :].multiply(self.item_similarity[:, i].T).sum(axis=1).getA1()[0]

    def rank(self, test_loader):
        rec_ids = None

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy()

            slims = np.expand_dims(self.interaction_matrix[us, :].todense(), axis=1) # batch * item_num -> batch * 1* item_num
            sims = self.item_similarity[cands_ids, :].transpose(0, 2, 1) # batch * cand_num * item_num -> batch * item_num * cand_num
            scores = np.einsum('BNi,BiM -> BNM', slims, sims).squeeze(axis=1) # batch * 1 * cand_num -> batch * cand_num
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[np.repeat(np.arange(len(rank_ids)).reshape(-1, 1), rank_ids.shape[1], axis=1), rank_ids]

            rec_ids = rank_list if rec_ids is None else np.vstack([rec_ids, rank_list])

        return rec_ids

    def full_rank(self, u):
        scores = self.interaction_matrix[u, :] @ self.item_similarity
        return np.argsort(-scores)[:, :self.topk]
    
    def rank_all(self):
        scores = self.interaction_matrix @ self.item_similarity
        return np.argsort(-scores)[:,:self.topk]