import os
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.sparse import csr_matrix

def create_matrix_and_mappings(dataframe: pd.DataFrame, scale) -> Tuple[csr_matrix, dict, dict]:
    """
    Create a CSR matrix and index mappings for users and items based on the given dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe containing user-item interactions.

    Returns:
        tuple[csr_matrix, dict, dict]: A tuple containing the CSR matrix representation of user-item interactions,
                                      the mapping of user indices to user values,
                                      and the mapping of item indices to item values.
    """

    # Extract unique users and items
    users = dataframe['user'].unique()
    items = dataframe['item'].unique()

    # Create user and item index mappings
    user_index = {user: i for i, user in enumerate(users)}
    item_index = {item: i for i, item in enumerate(items)}

    # Create CSR matrix
    row_indices = dataframe['user'].map(user_index)
    col_indices = dataframe['item'].map(item_index)
    values = np.ones(len(dataframe))*scale
    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(users), len(items)))

    # Create reverse mappings for user and item indices
    index_to_user = {i: user for user, i in user_index.items()}
    index_to_item = {i: item for item, i in item_index.items()}

    return matrix, index_to_user, index_to_item


def train_valid_split(train_df, num_seq, num_ran, random_seed=42):
    """
    Split the train dataframe into train and validation sets.

    Args:
        train_df (pandas.DataFrame): DataFrame containing the training data.
        num_seq (int): Number of data points per user to be included in the validation set.
        num_ran (int): Number of randomly selected previous data points per user to be included in the validation set.
        random_seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing train data and validation data as pandas DataFrames.
               The train data DataFrame contains the remaining data points after excluding the validation data.
               The validation data DataFrame consists of the last num_seq data points and num_ran randomly selected previous data points per user.
               Both DataFrames are sorted by the 'user' column.

    Raises:
        AssertionError: If the sum of num_seq and num_ran is not equal to 10.

    """
    # Set random seed
    np.random.seed(random_seed)

    total_num = 10
    assert np.isclose(num_seq + num_ran, total_num), "The sum of num_seq and num_ran should be equal to 10."


    # Extract the last num_seq data points per user to create valid data
    valid_data_last = train_df.groupby('user').tail(num_seq).copy()

    # Exclude valid data from train data
    train_data = train_df[~train_df.index.isin(valid_data_last.index)].copy()

    # Randomly select num_ran previous data points per user to create random_data
    random_data = train_data.groupby('user').apply(lambda x: x.sample(num_ran)).reset_index(drop=True)

    # Exclude random_data from train_data based on matching user and item values
    train_data = train_data[~train_data[['user', 'item']].apply(tuple, axis=1).isin(random_data[['user', 'item']].apply(tuple, axis=1))].copy()

    valid_data = pd.concat([valid_data_last, random_data], ignore_index=True)
    valid_data = valid_data[['user', 'item']].sort_values('user')

    return train_data, valid_data


def recall_at_10(true_df, pred_df):
    """
    Calculate Recall@10 metric for evaluating recommendation performance.

    Args:
        true_df (pandas.DataFrame): DataFrame containing true user-item interactions.
        pred_df (pandas.DataFrame): DataFrame containing predicted top-k items for each user.

    Returns:
        float: Mean Recall@10 score across all users.

    """
    # Create DataFrame of true interacted items for each user
    true_items = true_df.groupby('user')['item'].apply(set).reset_index(name='true_items')

    # Create DataFrame of predicted top-k items for each user
    pred_items = pred_df.groupby('user')['item'].apply(set).reset_index(name='pred_items')

    # Calculate recall@10 scores for each user
    recall_scores = []
    for _, row in true_items.iterrows():
        user = row['user']
        true_set = row['true_items']

        # Check if there are predicted items for the user
        pred_set = set(pred_items[pred_items['user'] == user]['pred_items'].values[0])
        intersection = true_set.intersection(pred_set)
        recall = len(intersection) / 10
        recall_scores.append(recall)

    # Calculate mean recall@10 across all users
    mean_recall = sum(recall_scores) / len(recall_scores)

    return mean_recall
