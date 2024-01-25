# CBMF (Class Balanced negative set by Maximum-Flow) and CBGS (Class Balanced negative set by Gibbs Sampling)

This repository contains files related to our work, CBMF and CBGS, which generate a set of negative enhancer-promoter interactions (EPIs) from 
a specified set of positive EPIs. 

Details of the directory structure of this repository are as follows.

```
CBOEP
├── cbmf.py
├── cbgs.py
├── data
└── output
```
`cbmf.py` and `cbgs.py` is the main execution file of CBMF and CBGS, respectively, given the positive EPIs set such as BENGI or TargetFinder.
Please place the required positive EPIs set for input under directory `data` (we have already placed BENGI and TargetFinder datasets as references).


# Datasets
CBMF and CBGS require a positive EPIs set as input to generate the new dataset.

We already have EPIs sets for BENGI and TargetFinder,
but if you want to generate the negative EPIs set from your EPIs set,
please create a freely directory in ```data```.

The positive EPIs set is a csv file and requires the following headers:  
| Header | Description |
| :---: | :---: |
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

# How to generate the new CBMF dataset
`cbmf.py` is the executable file to generate the CBMF dataset. 


## Requirements

| Library | Version |
| :---: | :---: |
| ```pandas``` | 1.3.4 |
| ```pulp``` | 2.6.0 |


## Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-input``` ||Path of the input EPI dataset.|
| ```-output``` ||Path of the output EPI dataset|
| ```-dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```-dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--concat``` |False|Whether or not to concatenate the CBOEP negative set with the positive set given as input. If not given, only the CBMF negative set will be output.|



## Execution example
```  
python cbmf.py \
-infile ./data/BENGI/GM12878.csv \
-outfile ./output/BENGI/dmax_2500000/GM12878.csv \
-dmax 2500000 \
-dmin 0 \
--concat
```

# How to generate the new CBGS dataset

`cbgs.py` is the executable file to generate the CBGS dataset. 

## Requirements

| Library | Version |
| :---: | :---: |
| ```pandas``` | 1.3.4 |
| ```pulp``` | 2.6.0 |


## Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-input``` ||Path of the input EPI dataset.|
| ```-output``` ||Path of the output EPI dataset|
| ```-dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```-dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
|```--T```|40,000|Number of sampling iteration|
| ```--concat``` |False|Whether or not to concatenate the CBOEP negative set with the positive set given as input. If not given, only the CBGS negative set will be output.|



## Execution example
```  
python cbgs.py \
-infile ./data/BENGI/GM12878.csv \
-outfile ./output/BENGI/dmax_2500000/GM12878.csv \
-dmax 2500000 \
-dmin 0 \
--concat
```







