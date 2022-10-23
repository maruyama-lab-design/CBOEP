from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import glob
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


outdir = os.path.join(os.path.dirname(__file__), "..", "output", "t-SNE")
feats, labels = [], []

research_name = "TF"
research_type_list = ["org", "mf"]

for research_type in research_type_list:
    # features
    data_dir = os.path.join(os.path.dirname(__file__), "..", "feat_before_sigmoid", f"{research_name}_{research_type}")
    file_list = glob.glob(os.path.join(data_dir, f"feats*.pkl"))
    for file in file_list:
        with open(file, mode='rb') as f:
            data = pickle.load(f)
            feats += data

    # labels
    data_dir = os.path.join(os.path.dirname(__file__), "..", "feat_before_sigmoid", f"{research_name}_{research_type}")
    file_list = glob.glob(os.path.join(data_dir, f"labels*.pkl"))
    for file in file_list:
        with open(file, mode='rb') as f:
            data = pickle.load(f)
            print(data)
            data = [f"positive" if i == [1.0] else f"{research_type}_negative" for i in data]
            labels += data



X = np.array(feats)
# y_item = set(labels)
y_item = ["positive", f"org_negative", f"mf_negative"]
y = np.array(labels)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

tsne = TSNE(n_components=2, random_state=0)
X_reduced = tsne.fit_transform(X)

color_list =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink"]
# color_dict = {}
# for i, label in enumerate(y_item):
#     color_dict[label] = color_list[i]

# color_dict["positive"] = "r"
# color_dict[f"org_negative"] = "g"
# color_dict[f"mf_negative"] = "b"

plt.figure(figsize=(13, 7))
for i, label in enumerate(y_item):
    print(f"plot label: {label}...")
    idx_list = (y == label)
    plt.scatter(X_reduced[idx_list, 0], X_reduced[idx_list, 1], label=label, color=color_list[i], s=5, alpha=0.5)
                # c=y, cmap='jet',
                # s=15, alpha=0.5)
plt.legend()
plt.axis('off')
plt.savefig(os.path.join(outdir, f"{research_name}.png"))