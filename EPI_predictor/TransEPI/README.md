# TransEPI

We have made some changes to the TransEPI code that had been published in https://github.com/biomed-AI/TransEPI/tree/main.

`cross_validation.py` is our main execution file of TransEPI.


<!-- ## Requirements
We have tested the work in the following environments.

| Library | Version |
| :---: | :---: |
|```python```|3.8.18|
|```pytorch```|2.2.0|
|```cudatoolkit```|11.8.0|
|```pandas```|2.0.3|
|```tqdm```||
|```matplotlib```|3.2.2|
|```tensorboard```|2.16.2| -->



## Preparation of genomic data

Before running,
download the genomic data folder from Onedrive (https://qu365-my.sharepoint.com/:f:/g/personal/maruyama_osamu_158_m_kyushu-u_ac_jp/EkiuCr1GzwNPolz9kCVqJu4BareEnjI_PpzwOVmtpIPqmA?e=Z8n8SM) and place it in the same location as `cross_validation.py`.

Then, please describe as follows in the variable "feats_config" in the `opt.json` file.

```
"feats_config": "[Path to the genomic data folder]/CTCF_DNase_6histone.500.json",
```




## Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```--train_mode``` ||If given, the model is trained.|
| ```--test_mode``` ||If given, the model is tested.|
| ```--train_EPI``` ||Path to an EPI dataset file to be used for training.|
| ```--test_EPI``` ||Path to an EPI dataset file to be used for test.|
| ```--train_cell``` ||Name of the cell line the training EPI dataset belongs to.|
| ```--test_cell``` ||Name of the cell line the test EPI dataset belongs to.|
| ```--model_dir``` |./models/|Directory to save trained models.|
| ```--pred_dir``` |./preds/|Directory to save prediction results.|
| ```--tensorboard_dir``` |./tensorboard/|Directory to save tensorborad logs.|
| ```--use_mask``` ||If given, only enhancer and promoter regions are used.|
| ```--use_weighted_bce``` ||If given, weights to the binary cross entropy based on the size ratio of the positive and negative EPIs are added.|
| ```--use_dist_loss``` ||If given, a loss function based on the distance between an enhancer and a promoter is added.|



## Execution example

```
python cross_validation.py \
--train_mode \
--test_mode \
--train_EPI ../../input_to_neg_generator/normalized_BENGI/GM12878.csv \
--test_EPI ../../input_to_neg_generator/normalized_BENGI/GM12878.csv \
--train_cell GM12878 \
--test_cell GM12878 \
--model_dir ./test/model/normalized_BENGI/GM12878/ \
--pred_dir ./test/prediction/normalized_BENGI/GM12878/ \
--tensorboard_dir ./test/tensorboard/normalized_BENGI/GM12878/ \
--use_weighted_bce 
```


# License

 ```
Copyright (c) 2021 Ken Chen 

Released under the MIT license 

https://github.com/biomed-AI/TransEPI/blob/main/LICENSE
 ```