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
ここまで

## `main` directory

This directory contains enhancer-promoter interaction data.  
CBOEP data can also be generated here.

Two types of existing EPI data
- BENGI
- TargetFinder

are contains in
- `/BENGI/original/{cell type}.csv`
- `/TargetFinder/original/{cell type}.csv`

respectively.

## How to generate new CBOEP dataset
`generate_CBOEP_dataset.py` is the executable file to generate CBOEP dataset.  

### Libraries 
---

| Library | Version |
| :---: | :---: |
| ```pandas``` | 1.3.4 |
| ```pulp``` | 2.6.0 |


### Argument
---

| argument | default value | description |
| :---: | :---: | ---- |
| ```--input``` |"BENGI"|Which positive interactions to use. Only "BENGI" or "TargetFinder" is accepted.|
| ```--dmax``` |2500000|Upper bound of enhancer-promoter distance for newly generated negative interactions.|
| ```--cell``` |"GM12878"|Cell type of the negative interactions; corresponding positive interactions are required.|

They can also be described in `CBOEP_opt.json`.  
Example of `CBOEP_opt.json`:  
```
{
    "input": "BENGI",
    "dmax": 3000000,
    "cell": "GM12878"
}
```

CBOEP dataset is generated in
- `pair_data/{--input}/CBOEP_{--dmax}/{--cell}.csv`




