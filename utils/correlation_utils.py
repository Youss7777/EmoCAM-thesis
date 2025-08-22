"""
Utility functions for correlation analysis of EmoNet results.

This module offers convenience functions for loading EmoNet outputs,
computing correlation matrices, plotting heatmaps and performing
statistical tests.  It integrates with other modules such as
``global_statistics`` and ``local_analysis`` to provide a unified
analysis pipeline.
"""

import matplotlib
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

matplotlib.use('MacOSX')

dist_list = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
             'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
             'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

def get_obj_vec(df_to_corr, obj_name):
    df2 = df_to_corr[['dir_image_path', 'detected_object']].copy()
    df2[obj_name] = (df2['detected_object'] == obj_name).astype(int)
    df3 = df2.groupby(by=['dir_image_path'], sort=False, as_index=False).sum(numeric_only=True)
    vec_obj = df3[obj_name].apply(lambda x: 1 if x > 1 else x)
    return vec_obj

def get_emo_vec(df_to_corr, emo_name, emo_label):
    df2 = df_to_corr[['dir_image_path', emo_label]].copy()
    df2[emo_name] = (df2[emo_label] == emo_name).astype(int)
    df3 = df2.groupby(by=['dir_image_path'], sort=False, as_index=False).sum(numeric_only=True)
    vec_emo = df3[emo_name].apply(lambda x: 1 if x > 1 else x)
    return vec_emo
def emo_emo_binary_df(df_to_corr, emo_to_corr1, emo_label1, emo_to_corr2, emo_label2):
    df = pd.DataFrame()
    for emo1 in emo_to_corr1:
        df = pd.concat([df, get_emo_vec(df_to_corr, emo1, emo_label1)], axis=1)
    for emo2 in emo_to_corr2:
        df = pd.concat([df, get_emo_vec(df_to_corr, emo2, emo_label2)], axis=1)
    return df

def emo_obj_binary_df(df_to_corr, emo_to_corr, emo_label, obj_to_corr):
    df = pd.DataFrame()
    for obj in obj_to_corr:
        df = pd.concat([df, get_obj_vec(df_to_corr, obj)], axis=1)
    for emo in emo_to_corr:
        df = pd.concat([df, get_emo_vec(df_to_corr, emo, emo_label)], axis=1)
    return df

# Simple Matching Coefficient : doesn't work
def smc_custom_metric(u, v):
    matches = np.sum(u == v)
    total = len(u)
    return 1 - matches / total  # pdist expects a distance, so we return 1 - SMC
def smc_correlation_matrix(df):
    bool_array = df.to_numpy().astype(bool)
    smc_distances = pdist(bool_array.T, metric=smc_custom_metric)
    smc_similarity_matrix = squareform(smc_distances)
    return pd.DataFrame(1 - smc_similarity_matrix, index=df.columns, columns=df.columns)

# Dice : doesn't work
def dice_custom_metric(u, v):
    intersection = np.sum(np.logical_and(u, v))
    total = np.sum(u) + np.sum(v)
    return 2 * intersection / total if total != 0 else 0
def dice_correlation_matrix(df):
    bool_array = df.to_numpy().astype(bool)
    dice_distances = pdist(bool_array.T, metric=dice_custom_metric)
    dice_similarity_matrix = 1 - squareform(dice_distances)
    return pd.DataFrame(dice_similarity_matrix, index=df.columns, columns=df.columns)

# Cosine : doesn't work
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product != 0 else 0
def cosine_correlation_matrix(df):
    bool_array = df.to_numpy().astype(bool)
    cosine_distances = pdist(bool_array.T, metric=cosine_similarity)
    cosine_similarity_matrix = 1 - squareform(cosine_distances)
    return pd.DataFrame(cosine_similarity_matrix, index=df.columns, columns=df.columns)

# Jaccard: good
def jaccard_correlation_matrix(df):
    # Convert DataFrame to boolean numpy array
    bool_array = df.to_numpy().astype(bool)
    # Compute pairwise Jaccard distances
    jaccard_distances = pdist(bool_array.T, metric='jaccard')
    # Convert distances to square form matrix
    jaccard_similarity_matrix = 1 - squareform(jaccard_distances)
    # Convert result to DataFrame
    return pd.DataFrame(jaccard_similarity_matrix, index=df.columns, columns=df.columns)


# Rajski : good
def rajski_distance(vec1, vec2):    # same as mutual information?
    numerator = np.sum(np.logical_and(vec1, ~vec2))
    denominator = np.sum(vec1)
    return numerator / denominator if denominator != 0 else 0


def rajski_correlation_matrix(df):
    bool_array = df.to_numpy().astype(bool)
    rajski_distances = pdist(bool_array.T, metric=rajski_distance)
    rajski_similarity_matrix = 1 - squareform(rajski_distances)
    return pd.DataFrame(rajski_similarity_matrix, index=df.columns, columns=df.columns)


def binary_vec_correlation_matrix(df, method=None):
    if method is None:
        method = 'jaccard'
    # convert to bool
    bool_array = df.to_numpy().astype(bool)
    # compute distances (transpose since cols = features and rows = samples)
    distances = pdist(bool_array.T, metric=method)
    # normalization
    #min_dist = distances.min()
    #max_dist = distances.max()
    #normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    # convert back to matrix and from dissimilarity to similarity
    similarity_matrix = 1 - squareform(distances)
    return pd.DataFrame(similarity_matrix, index=df.columns, columns=df.columns)

