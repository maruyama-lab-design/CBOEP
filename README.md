# Generators of Negative enhancer-promoter pairs, CBMF (Class Balanced negative set by Maximum-Flow) and CBGS (Class Balanced negative set by Gibbs Sampling)

This repository contains files related to our methods, CBMF and CBGS, which generate a set of negative enhancer-promoter interactions (EPIs) from a set of positive EPIs. 
The implementations of the methods are `cbmf.py` and `cbgs.py` in the top directory. 

# Datasets
CBMF and CBGS require a positive EPIs set as input to generate a negative dataset.

The resulting negative sets generated by the methods from BENGI positive sets and 
positive sets used in the work of TargetFinder are stored in 
directory ```input_to_neg_generator```.

If you want to generate negative EPI sets from a positive EPI set,
please note that the positive set should be in the csv format with the following columns:

| Column | Description |
| :---: | --- |
| ```label``` | ```1``` for positive EPI, ```0``` for negative EPI |
| ```enhancer_distance_to_promoter``` | Distance between the enhancer and the promoter |
| ```enhancer_chrom``` | Chromosome number of the enhancer |
| ```enhancer_start``` | Start position of the enhancer |
| ```enhancer_end``` | End position of the enhancer |
| ```enhancer_name``` | Name of the enhancer, such as `GM12878\|chr16:88874-88924` |
| ```promoter_chrom``` | Chromosome number of the promoter |
| ```promoter_start``` | Start position of the promoter |
| ```promoter_end``` | End position of the promoter |
| ```promoter_name``` | Name of the promoter, such as `GM12878\|chr16:103009-103010`|


## CBMF Requirements
We have tested the work in the following environments.

| Library | Version |
| :---: | :---: |
|```python```|3.8.0|
| ```numpy``` |1.18.1|
| ```pandas``` |1.0.1|
| ```pulp``` | 2.8.0 |


## CBMF Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-input``` ||Path to the input EPI dataset.|
| ```-output``` ||Path to the output EPI dataset|
| ```-dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```-dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--concat``` |False|Whether or not to concatenate the CBMF negative set with the positive set given as input. If not given, only the CBMF negative set will be output.|



## CBMF Execution example
```  
python cbmf.py \
-infile ./input_to_neg_generator/normalized_BENGI/GM12878.csv \
-outfile ./output_from_neg_generator/BENGI-P_CBMF-N/GM12878.csv \
-dmax 2500000 \
-dmin 0 \
--concat
```


## CBGS Requirements

We have tested the work in the following environments.

| Library | Version |
| :---: | :---: |
|```python```|3.8.0|
| ```numpy``` |1.18.1|
| ```pandas``` |1.0.1|
| ```matplotlib``` | 3.2.2 |

## CBGS Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-input``` ||Path to the input EPI dataset.|
| ```-output``` ||Path to the output EPI dataset|
| ```-dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```-dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
|```--T```|40,000|Number of sampling iteration|
| ```--concat``` ||If given, the CBGS negative set is concatenated with the positive set given as input. If not given, only the CBGS negative set will be output.|
|```--make_fig```||If given, a figure which shows plots of the mean of positive/negative class imbalance of all enhancers and promoters for each sampling iteration is made.|
|```--out_figfile```||If ```--make_fig``` is given, a figure is saved in this path.|


## CBGS Execution example
```  
python cbgs.py \
-infile ./input_to_neg_generator/normalized_BENGI/GM12878.csv \
-outfile ./output_from_neg_generator/BENGI-P_CBGS-N/GM12878.csv \
-dmax 2500000 \
-dmin 0 \
--concat \
--make_fig \
--out_figfile ./output_from_neg_generator/BENGI-P_CBGS-N/GM12878.png
```







