# TargetFinder

We have made some changes to the TargetFinder code that had been published in https://github.com/shwhalen/targetfinder/tree/master.


`cross_validation.py` is the main execution file of TargetFinder. 

<!-- ## Requirements
We have tested the work in the following environments.

| Library | Version |
| :---: | :---: |
|```python```|3.8.0|
|```pandas```|2.0.3|
|```joblib```|1.3.2|
|```scipy```|1.10.1|
|```scilit-learn```|1.3.0| -->


## Data Preprocessing


Before running,
download the folder ```input_features``` from [Onedrive](https://qu365-my.sharepoint.com/:f:/g/personal/maruyama_osamu_158_m_kyushu-u_ac_jp/Ev1P9trUWRhDleRyd1xn8EMBzZxch881tSOQWLRSWZHEzg?e=gqBFZh) and place it in the same location as `cross_validation.py`.

Then, this genomic features must be added to the EPI dataset using `preprocess.py`.
The following command adds genomic data to EPI data whose file path is given in `-i`.

```
python preprocess.py \
-i  ../../input_to_neg_generator/normalized_BENGI/GM12878.csv \
-o  ../../featured_EPI/normalized_BENGI/GM12878.csv \
--cell GM12878 \
--use_window  \
--data_split 20
```

If the program does not work correctly due to insufficient memory,
increasing the value of ```--data_split``` can solve the memory shortage problem.




## Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```--train_EPI``` ||Path to an EPI dataset file (added genomic features) to be used for training.|
| ```--test_EPI``` ||Path to an EPI dataset file (added genomic features) to be used for test.|
| ```--pred_dir``` |./preds/|Directory to save prediction results.|
| ```--use_window``` ||If not given, only enhancer and promoter regions are used.|



## Execution example

```
python cross_validation.py \
--train_EPI ../../featured_EPI/normalized_BENGI/GM12878.csv \
--test_EPI ../../featured_EPI/normalized_BENGI/GM12878.csv \
--pred_dir ./prediction/normalized_BENGI/GM12878/ \
--use_window
```

# License

GNU General Public License v3.0 is used in https://github.com/shwhalen/targetfinder/blob/master/LICENSE.

In our script,
there are two major differences from the original TargetFinder.

One is that the script to add genomic features to EPI data is in ```generate_training.py``` in the TargetFinder repository, and we found a problem that some EPI data did not fit in memory, so we improved the script and replaced it with ```preprocess.py```.

Another is the addition of ```cross_validation.py``` to perform chromosome-wise cross validation using a gradient boosting decision tree.


