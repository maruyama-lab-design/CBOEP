# CBOEP (Class Balanced Occurrence of Enhancers and Promoters)

This repository contains files related to our work, CBOEP, which generates a set of negative enhancer-promoter interactions (EPIs) from 
a specified set of positive EPIs. 




This repository contains three directories, `main`, `TransEPI`, and `TargetFinder`.  

In the directory of 'main',
we include the BENGI dataset ,TargetFinder dataset and CBOEP dataset (d_max=2.5M) for each of them.
In addition to this, you can generate the CBOEP dataset of any d_max.  

In `/TransEPI` or `/TargetFinder`,
you can perform EPI prediction with existing EPI predictors, "TransEPI" or "TargetFinder".  
For more information on each predictor, see the following papers.  
- TransEPI: [Capturing large genomic contexts for accurately predicting enhancer-promoter interactions](https://academic.oup.com/bib/article-abstract/23/2/bbab577/6513727?redirectedFrom=fulltext&login=false)  
- TargetFinder: [Enhancer-promoter interactions are encoded by complex genomic signatures on looping chromatin](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4910881/)  


Details of the directory structure are as follows.

```
CBOEP
├── main
│   ├── pair_data
│   └── generate_CBOEP_dataset.py  
├── TransEPI
│   ├──   
│   └── main.py  
└── TargetFinder
    ├──   
    └── main.py  
```


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




