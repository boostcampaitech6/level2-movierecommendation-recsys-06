import numpy as np
import scipy.sparse as sp
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count

class EaseModel:     
    def __init__(self, _lambda):
        self.B = None
        self._lambda = _lambda
        
    def train(self, X):

        # Compute the G matrix
        G = X.T @ X  # item_num * item_num
        G += self._lambda * sp.identity(G.shape[0])  # Regularization term
        G = G.todense()  # Convert to a dense matrix

        # Compute the P matrix
        P = np.linalg.inv(G)

        # Compute the B matrix (item similarity matrix)
        self.B = -P / np.diag(P)  # equation 8 in the paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
        np.fill_diagonal(self.B, 0.)  # Set diagonal elements to zero

        # Store the item similarity matrix and interaction matrix
        self.item_similarity = np.array(self.B)  # item_num * item_num
        self.interaction_matrix = X  # user_num * item_num

    def predict(self, train, users, items, k):
        items = self.item_enc.transform(items)
        dd = train.loc[train.user.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.item)
        dd['cu'] = self.user_enc.transform(dd.user)
        g = dd.groupby('cu')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        df['item'] = self.item_enc.inverse_transform(df['item'])
        df['user'] = self.user_enc.inverse_transform(df['user'])
        return df

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['ci'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user": [user] * len(res),
                "item": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        return r
    
    def forward(self, user_row):
        
        return user_row @ self.B
    
