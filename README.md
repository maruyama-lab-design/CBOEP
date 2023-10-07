# CBOEP (Class Balanced Occurrence of Enhancers and Promoters)

This repository contains files related to our work, CBOEP, which generates a set of negative enhancer-promoter interactions (EPIs) from 
a specified set of positive EPIs. 

Details of the directory structure of this repository are as follows.

```
CBOEP
├── cboep.py
├── data
└── output
```
`cboep.py` is the main execution file to generate the negative EPIs set, given the positive EPIs set such as BENGI or TargetFinder.
Please place the required positive EPIs set for input under directory `data` (we have already placed BENGI and TargetFinder datasets as references).

# Requirements

| Library | Version |
| :---: | :---: |
| ```pandas``` | 1.3.4 |
| ```pulp``` | 2.6.0 |

# Datasets
CBOEP requires a positive EPIs set as input to generate the new dataset.

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

# How to generate the new CBOEP dataset
`cboep.py` is the executable file to generate the CBOEP dataset. 

## Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-input``` ||Which positive EPIs set to use.|
| ```-output``` ||Path of the output EPI dataset|
| ```-dmax``` |2500000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```-dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--aplha``` |1.0||
| ```--concat``` ||Whether or not to concatenate the CBOEP negative set with the positive set given as input.
If not given, only the CBOEP negative set will be output.|



## Execution example
```  
python cboep.py \
-infile ./data/BENGI/GM12878.csv \
-outfile ./output/BENGI/GM12878_1.5.csv \
-dmax 2500000 \
-dmin 0 \
--alpha 1.5
```









