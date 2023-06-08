# CBOEP (Class Balanced Occurrence of Enhancers and Promoters)

This repository contains files related to our work, CBOEP, which generates a set of negative enhancer-promoter interactions (EPIs) from 
a specified set of positive EPIs. 

Details of the directory structure of this repository are as follows.

```
CBOEP
├── cboep.py
├── data
├── output
└── EPIpredictors
    ├── TransEPI  
    └── TargetFinder 
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
The positive EPIs set is a csv file and requires the following headers:  
| Header | Description |
| :---: | :---: |
| ```label``` | ```1``` for positive EPI, ```0``` for negative EPI |
| ```enhancer_distance_to_promoter``` | Distance between the enhancer and the promoter |
| ```enhancer_chrom``` | Chromosome number of the enhancer |
| ```enhancer_start``` | Start position of the enhancer |
| ```enhancer_end``` | End position of the enhancer |
| ```enhancer_name``` | Name composed of the cell line and region position to which the enhancer belongs, such as `GM12878|chr16:88874-88924` |
| ```promoter_chrom``` | Chromosome number of the promoter |
| ```promoter_start``` | Start position of the promoter |
| ```promoter_end``` | End position of the promoter |
| ```promoter_name``` | Name composed of the cell line and region position to which the promoter belongs, such as `GM12878|chr16:103009-103010`|

# How to generate the new CBOEP dataset
`cboep.py` is the executable file to generate the CBOEP dataset. 

## Argument
---

| Argument | Default value | Description |
| :---: | :---: | ---- |
| ```-input``` |"BENGI"|Which positive interactions to use. Only "BENGI" or "TargetFinder" is accepted.|
| ```-dmax``` |2500000|Upper bound of enhancer-promoter distance for newly generated negative interactions.|
| ```-cell``` |"GM12878"|Cell type of the negative interactions; the corresponding positive EPIs set are required.|

## Execution example
```  
python cboep.py -input BENGI -dmax 5000000 -cell K562
```

CBOEP dataset is generated in `output/{-input}/dmax_{-dmax}/{-cell}.csv`








