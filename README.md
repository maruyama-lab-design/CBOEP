# NIMF (Negative Interaction by Maximum Flow)

The codes and datasets for "paper url".
This repository contains three directories, `/pair_data`, `/TransEPI`, and `/TargetFinder`.  

In `/pair_data`,
we include the BENGI dataset ,TargetFinder dataset and NIMF dataset (d_max=2.5M) for each of them.
In addition to this, you can generate the NIMF dataset of any d_max.  

In `/TransEPI` or `/TargetFinder`,
you can perform EPI prediction with existing EPI predictors, "TransEPI" or "TargetFinder".  
For more information on each predictor, see the following papers.  
- TransEPI: [Capturing large genomic contexts for accurately predicting enhancer-promoter interactions](https://academic.oup.com/bib/article-abstract/23/2/bbab577/6513727?redirectedFrom=fulltext&login=false)  
- TargetFinder: [Enhancer-promoter interactions are encoded by complex genomic signatures on looping chromatin](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4910881/)  


Details of the directory structure are as follows.

```
NIMF
├── pair_data
│   ├── BENGI
│   ├── TargetFinder
│   └── generate_NIMF_dataset.py  
├── TransEPI
│   ├──   
│   └── main.py  
└── TargetFinder
    ├──   
    └── main.py  
```


## `pair_data` directory

This directory contains enhancer-promoter interaction data.  
NIMF data can also be generated here.

Two types of existing EPI data
- BENGI
- TargetFinder

are contains in
- `/BENGI/original/{cell type}.csv`
- `/TargetFinder/original/{cell type}.csv`

respectively.

## How to generate new NIMF dataset
`generate_NIMF_dataset.py` is the executable file to generate NIMF dataset.  

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
| ```--data``` |"BENGI"|Which positive interactions to use. Only "BENGI" or "TargetFinder" is accepted.|
| ```--NIMF_max_d``` |2500000|Upper bound of enhancer-promoter distance for newly generated negative interactions.|
| ```--cell_type``` |"GM12878"|Cell type of the negative interactions; corresponding positive interactions are required.|

They can also be described in `NIMF_opt.json`.  
Example of `NIMF_opt.json`:  
```
{
    "data": "BENGI",
    "NIMF_max_d": 3000000,
    "cell_type": "GM12878"
}
```

NIMF dataset is generated in
- `/{--data}/NIMF_{--NIMF_max_d}/{--cell_type}.csv`

## `TransEPI` directory


## `TargetFinder` directory



