import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

import json


json_dict = {}

n_fold = 10

for i in range(n_fold):
    json_dict[f"fold{i}"] = {
        "train_chroms": [],
        "valid_chroms": [],
        "test_chroms": []
    }

all_chroms = np.array([f"chr{i}" for i in list(range(1, 23)) + ["X"]])

kf = KFold(n_splits=n_fold)
for i, (train_idx, tmp_idx) in enumerate(kf.split(all_chroms)):
    train_chroms = all_chroms[train_idx]
    tmp_chroms = all_chroms[tmp_idx]
    valid_chroms, test_chroms, _, _ = train_test_split(tmp_chroms, tmp_chroms, train_size=0.5)

    print(f"___fold {i}___")
    print(f"_train chroms_")
    print(train_chroms)
    print(f"_valid chroms_")
    print(valid_chroms)
    print(f"_test chroms_")
    print(test_chroms)
    print("\n")

    json_dict[f"fold{i}"]["train_chroms"] = train_chroms.tolist()
    json_dict[f"fold{i}"]["valid_chroms"] = valid_chroms.tolist()
    json_dict[f"fold{i}"]["test_chroms"] = test_chroms.tolist()


path = './chromosome_split_opt.json'
json_file = open(path, mode="w")
json.dump(json_dict, json_file)
json_file.close()