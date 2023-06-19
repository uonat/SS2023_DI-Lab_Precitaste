import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

current_directory = os.getcwd()
sys.path.insert(0, current_directory)   # get current dir and insert to sys.path pos 0

from notebooks.utils.dino_v2 import read_pred_objects_json

def tsne_visualization(output_dir, vis_num, use_global_label=False):

    global_or_fine = 'global_labels' if use_global_label else 'fine_labels'

    path_to_val_pred_json_file = Path("dataset") / "dinov2_processed_data" / "vith_res_val_processed_pred_objects_2.json"
    all_objects = read_pred_objects_json(str(path_to_val_pred_json_file))
    to_be_visualized_objects = [obj for obj in all_objects if obj.gt_label is not None and obj.pred_features is not None][:vis_num]
    features = np.array([obj.pred_features[0] for obj in to_be_visualized_objects])
    labels = np.array([obj.gt_label for obj in to_be_visualized_objects])
    if use_global_label:
        labels = [label.split('_', 1)[1] for label in labels]

    # get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(features)
    print(tsne_result.shape)

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels})
    fig, ax = plt.subplots(figsize=(12,8))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=100)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    output_dir = os.path.join(output_dir, "output", "tsne", "dinov2")
    os.makedirs(output_dir, exist_ok=True)
  
    output_name = os.path.join(output_dir, f"{global_or_fine}_{vis_num}.png")
    plt.title(global_or_fine + ' ' + str(tsne_result.shape[0]) + 'samples')
    # plt.show()
    plt.savefig(output_name)

if __name__ == 'main':
    # args: output_dir, vis_num, (optional: use_global_label=True/False)
    tsne_visualization(current_directory, 1000)
