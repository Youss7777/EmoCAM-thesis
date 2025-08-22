"""
Compute and visualise global statistics from EmoNet outputs.

This module defines a collection of functions for analysing the
outputs of EmoNet across a dataset.  It can compute correlations,
plot heatmaps, perform statistical tests and generate summary
figures.  See the accompanying thesis for a detailed description of
the statistical methods applied.
"""

import pandas as pd
import models.emonet as emonet
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scipy as sp
import os
import numpy as np
import utils.correlation_utils as corr

findingemo_emotions_list = ['Acceptance', 'Admiration', 'Amazement', 'Anger', 'Annotance', 'Anticipation', 'Apprehension',
                            'Boredom', 'Digust', 'Distraction', 'Ecstast', 'Fear', 'Grief', 'Interest', 'Joy', 'Loathing', 'Pensiveness', 'Rage', 'Serenity',
                            'Surprise', 'Terror', 'Trust', 'Vigilance']
object_to_corr_emo_obj = ['Person', 'Human face', 'Human eye', 'Human nose', 'Human mouth', 'Human head', 'Clothing',
                          'Sports equipment', 'Footwear', 'Plant', 'Flower', 'Animal', 'Food', 'Drink', 'Furniture',
                          'Weapon', 'Wheel', 'Building', 'Personal Care', 'Vehicle', 'Tableware', 'Kitchenware',
                          'Pillow', 'Jeans']
object_to_corr_ann_obj = ['Person', 'Human face', 'Human eye', 'Human nose', 'Human mouth', 'Human head', 'Clothing',
                          'Sports equipment', 'Footwear', 'Plant', 'Flower', 'Tree', 'Animal', 'Food', 'Drink',
                          'Furniture', 'Weapon', 'Wheel', 'Building', 'Personal Care', 'Girl', 'Vehicle']

def get_title_corr(method: str) -> str:
    titles = {
        "braycurtis": "Bray-Curtis",
        "canberra": "Canberra",
        "chebyshev": "Chebyshev",
        "cityblock": "City-Block",
        "correlation": "Pearson",
        "cosine": "Cosine",
        "dice": "Dice",
        "euclidean": "Euclidean",
        "hamming": "Hamming",
        "jaccard": "Jaccard",
        "jensenshannon": "Jensen-Shannon",   # fixed typo
        "kulczynski1": "Kulczynski",
        "mahalanobis": "Mahalanobis",
        "matching": "Simple Matching",
        "minkowski": "Minkowski",
        "rogerstanimoto": "Rogers-Tanimoto",
        "russellrao": "Russell-Rao",
        "seuclidean": "Standardized Euclidean Distance",
        "sokalmichener": "Sokal-Michener",
        "sokalsneath": "Sokal-Sneath",
        "sqeuclidean": "Squared Euclidean Distance",
        "yule": "Yule",
    }

    title_corr = titles.get(method, method if method else "no title")
    return f"{title_corr} correlation"

def get_dir_image_path(file_path):
    return os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path)


def get_annotations_df(config):
    """
    Merge the FindingEmo annotations with the outputs of EmoNet.
    """
    df_annotations = pd.read_csv(config.PATH_ANNOTATIONS)
    # modify annotations header to distinguish from EmoNet outputs
    df_annotations = df_annotations.rename(columns={'user': 'ann_user', 'image_path': 'ann_original_image_path',
                                                    'reject': 'ann_reject', 'age': 'age_group',
                                                    'valence': 'ann_valence', 'arousal': 'ann_arousal',
                                                    'emotion': 'ann_emotion', 'dec_factors': 'ann_dec_factors',
                                                    'ambiguity': 'ann_ambiguity',
                                                    'fmri_candidate': 'ann_fmri_candidate',
                                                    'datetime': 'ann_datetime'})
    # add 'dir_image_path' as path containing only folder name and file name
    df_annotations['dir_image_path'] = df_annotations['ann_original_image_path'].apply(get_dir_image_path)
    return df_annotations


def save_dictionary(dico, name):
    with open(f"{name}.json", "w") as file:
        json.dump(dico, file)
    print(f'{name} saved successfully.')


def remove_outliers(df, col_name, freq):
    """
    Remove the detected objects which occurring frequency is below 'freq'
    """
    v = df[col_name].value_counts(normalize=True)
    df = df[df[col_name].isin(v.index[v.gt(freq)])]
    return df


def remove_instances(df, col_name, instances):
    """
    Remove all instances from 'col_name' column of dataframe df
    """
    for inst in instances:
        df = df[(df[col_name] != inst)]
    return df


class GlobalStatistics:

    def __init__(self, config, obj_importance_thres, emo_conf_thres, obj_conf_thres, ann_ambiguity_thres,
                 method):
        self.method = method
        self.config = config
        self.ann = get_annotations_df(config)
        self.obj_importance_thres = obj_importance_thres
        self.emo_conf_thres = emo_conf_thres
        self.obj_conf_thres = obj_conf_thres
        self.ann_ambiguity_thres = ann_ambiguity_thres
        self.emonet_outputs, self.yolo_outputs, self.yolo_ann_outputs, self.emonet_ann_outputs = self.process_df()
        self.yolo_outputs_filtered, self.yolo_ann_outputs_filtered, self.emonet_ann_outputs_filtered = self.filter_df()

    def process_df(self):
        # load & remove surplus images
        emonet_outputs = pd.read_csv(self.config.PATH_MODEL_OUTPUTS+'/outputs_'+self.method+'/emonet_outputs_'+self.method+'.csv')
        emonet_outputs = pd.merge(self.ann['dir_image_path'], emonet_outputs, how='left', on='dir_image_path')
        yolo_outputs = pd.read_csv(self.config.PATH_MODEL_OUTPUTS+'/outputs_'+self.method+'/yolo_outputs_'+self.method+'.csv')
        yolo_outputs = pd.merge(self.ann['dir_image_path'], yolo_outputs, how='left', on='dir_image_path')

        # some merging for emo_obj and emo_ann analyses
        yolo_ann_outputs = pd.merge(yolo_outputs, self.ann, on=["dir_image_path"], how='left')
        emonet_ann_outputs = pd.merge(emonet_outputs, self.ann, on=["dir_image_path"], how='left')

        # drop objects that are detected multiple times in the same image
        yolo_outputs = yolo_outputs.drop_duplicates(subset=['dir_image_path', 'emonet_emotion', 'detected_object'],
                                                    keep='first')
        yolo_ann_outputs = yolo_ann_outputs.drop_duplicates(subset=['dir_image_path', 'ann_emotion', 'detected_object'],
                                                            keep='first')

        return emonet_outputs, yolo_outputs, yolo_ann_outputs, emonet_ann_outputs

    def filter_df(self):
        # apply pre-filtering
        yolo_outputs_filtered = self.yolo_outputs[(self.yolo_outputs["emonet_emotion_conf"] > self.emo_conf_thres) &
                                                  (self.yolo_outputs["object_confidence"] > self.obj_conf_thres) &
                                                  (self.yolo_outputs["object_importance"] > self.obj_importance_thres)]
        yolo_ann_outputs_filtered = self.yolo_ann_outputs[
            (self.yolo_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
            (self.yolo_ann_outputs["object_confidence"] > self.obj_conf_thres)]
        emonet_ann_outputs_filtered = self.emonet_ann_outputs[
            (self.emonet_ann_outputs["ann_ambiguity"] < self.ann_ambiguity_thres) &
            (self.emonet_ann_outputs["emonet_emotion_conf"] > self.emo_conf_thres)]
        return yolo_outputs_filtered, yolo_ann_outputs_filtered, emonet_ann_outputs_filtered

    def get_emo_obj_df(self):
        return self.yolo_outputs_filtered[['dir_image_path', 'emonet_emotion', 'detected_object']]

    def get_ann_obj(self):
        return self.yolo_ann_outputs_filtered[['dir_image_path', 'ann_emotion', 'detected_object']]

    def get_emo_ann_df(self):
        return self.emonet_ann_outputs_filtered[['dir_image_path', 'emonet_emotion', 'ann_emotion']]

    def get_aro_df(self):
        return self.emonet_ann_outputs_filtered[['emonet_arousal', 'ann_arousal']]

    def get_val_df(self):
        return self.emonet_ann_outputs_filtered[["emonet_valence", "ann_valence"]]

    def plot_scatter_size_plot(self, df, col1, col2):
        c = pd.crosstab(df[col1], df[col2]).stack().reset_index(name='C')
        c.plot.scatter(col1, col2, s=c.C, figsize=(11, 8), fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, correlation_matrix, method, cmap=None):
        #mask = np.triu(np.ones_like(correlation_matrix))  # mask for triangular matrix only
        plt.figure(figsize=(9.3, 7))
        if method in ['correlation', 'yule']:
            cmap = 'coolwarm'
        m = sns.heatmap(correlation_matrix, annot=False, xticklabels=True, yticklabels=True, cmap=cmap,
                        linewidths=0, rasterized=True)
        x_labels = m.get_xticklabels()
        y_labels = m.get_yticklabels()
        m.set_xticklabels(x_labels, fontsize=7)
        m.set_yticklabels(y_labels, fontsize=7)
        plt.title(get_title_corr(method))
        plt.tight_layout()
        plt.show()

    def analyze(self, analysis_type: str):
        analysis_map = {
            "emo_obj": (
                self.analysis_emo_obj,
                dict(emo_to_corr=emonet.EmoNet.EMOTIONS,
                     obj_to_corr=object_to_corr_emo_obj,
                     method="dice")
            ),
            "ann_obj": (
                self.analysis_ann_obj,
                dict(emo_to_corr=findingemo_emotions_list,
                     obj_to_corr=object_to_corr_ann_obj,
                     method="dice")
            ),
            "emo_ann": (
                self.analysis_emo_ann,
                dict(emo_to_corr1=emonet.EmoNet.EMOTIONS,
                     emo_to_corr2=findingemo_emotions_list,
                     method="dice")
            ),
            "valence": (self.analysis_valence, {}),
            "arousal": (self.analysis_arousal, {}),
            "val_aro": (self.analysis_val_aro, {}),
        }

        if analysis_type not in analysis_map:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        func, kwargs = analysis_map[analysis_type]
        func(**kwargs)

    def analysis_emo_obj(self, emo_to_corr, obj_to_corr, method=None):
        # analysis 1 : detected object (Yolo & GradCam) vs predicted emotion (EmoNet)
        # get data
        #emo_obj_df = gs.get_emo_obj_df()
        # remove specific instances
        #emo_obj_df = remove_instances(emo_obj_df, 'detected_object', ['Person', 'Clothing'])
        # scatter plot
        #gs.plot_scatter_size_plot(emo_obj_df, "emonet_emotion", "detected_object")
        # correlation matrix
        df = corr.emo_obj_binary_df(df_to_corr=self.get_emo_obj_df(), emo_to_corr=emo_to_corr,
                                                 emo_label='emonet_emotion', obj_to_corr=obj_to_corr)
        correlation_matrix = corr.binary_vec_correlation_matrix(df, method=method)
        # choose specific method for similarity measure
        #gs.plot_correlation_heatmap(correlation_matrix, method=method)
        return correlation_matrix


    def analysis_ann_obj(self, emo_to_corr, obj_to_corr, method=None):
        # analysis 1 : detected object (Yolo & GradCam) vs predicted emotion (EmoNet)
        #ann_obj_df = gs.get_ann_obj()
        # remove specific instances
        #ann_obj_df = remove_instances(ann_obj_df, 'detected_object', ['Person', 'Clothing'])
        # scatter plot
        #gs.plot_scatter_size_plot(ann_obj_df, "ann_emotion", "detected_object")
        # correlation matrix
        df = corr.emo_obj_binary_df(df_to_corr=self.get_ann_obj(), emo_to_corr=emo_to_corr,
                                                 emo_label='ann_emotion', obj_to_corr=obj_to_corr)
        correlation_matrix = corr.binary_vec_correlation_matrix(df, method=method)
        # plot
        self.plot_correlation_heatmap(correlation_matrix, method=method)


    def analysis_emo_ann(self, emo_to_corr1, emo_to_corr2, method=None):
        # analysis 2 : predicted emotion (EmoNet) vs annotated emotion (ANN)
        #emo_ann_df = gs.get_emo_ann_df()
        # scatter plot
        #gs.plot_scatter_size_plot(emo_ann_df, "emonet_emotion", "ann_emotion")
        # correlation matrix
        df = corr.emo_emo_binary_df(df_to_corr=self.get_emo_ann_df(), emo_to_corr1=emo_to_corr1,
                                                 emo_label1='emonet_emotion', emo_to_corr2=emo_to_corr2,
                                                 emo_label2='ann_emotion')
        correlation_matrix = corr.binary_vec_correlation_matrix(df, method=method)

        # plot
        self.plot_correlation_heatmap(correlation_matrix, method=method)


    def analysis_valence(self):
        # analysis 4 : predicted valence (EmoNet) vs annotated valence (ANN)
        val_emonet_ann_df = self.get_val_df()
        # plotting
        plt.scatter(x=val_emonet_ann_df["emonet_valence"], y=val_emonet_ann_df["ann_valence"], color='#4169E1')
        # plot correlation matrix (one coefficient only here)
        print(val_emonet_ann_df.corr(method="spearman"))
        print(sp.stats.pearsonr(val_emonet_ann_df["emonet_valence"], val_emonet_ann_df["ann_valence"]))
        print(sp.stats.spearmanr(val_emonet_ann_df["emonet_valence"], val_emonet_ann_df["ann_valence"]))
        plt.xlabel('emonet_valence')
        plt.ylabel('ann_valence')
        plt.tight_layout()
        plt.show()


    def analysis_arousal(self):
        # analysis 5 : predicted arousal (EmoNet) vs annotated arousal (ANN)
        aro_emonet_ann_df = self.get_aro_df()
        # scatter plot
        plt.scatter(x=aro_emonet_ann_df["emonet_arousal"], y=aro_emonet_ann_df["ann_arousal"], color='#4169E1')
        # plot correlation matrix (one coefficient only here)
        print(aro_emonet_ann_df.corr(method="spearman"))
        print(sp.stats.pearsonr(aro_emonet_ann_df["emonet_arousal"], aro_emonet_ann_df["ann_arousal"]))
        print(sp.stats.spearmanr(aro_emonet_ann_df["emonet_arousal"], aro_emonet_ann_df["ann_arousal"]))
        plt.xlabel('emonet_arousal')
        plt.ylabel('ann_arousal')
        plt.tight_layout()
        plt.show()

    def analysis_val_aro(self):
        mask = np.triu(np.ones_like(self.emonet_ann_outputs_filtered[['emonet_valence', 'ann_valence', 'emonet_arousal', 'ann_arousal']].corr(method='pearson')))
        m = sns.heatmap(self.emonet_ann_outputs_filtered[['emonet_valence', 'ann_valence', 'emonet_arousal', 'ann_arousal']].corr(method='pearson'),
                    annot=True, xticklabels=True, yticklabels=True, mask=mask, cmap='coolwarm')
        x_labels = m.get_xticklabels()
        y_labels = m.get_yticklabels()
        m.set_xticklabels(x_labels, fontsize=11)
        m.set_yticklabels(y_labels, fontsize=11)
        plt.title('Pearson correlation')
        plt.tight_layout()
        plt.show()
