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




