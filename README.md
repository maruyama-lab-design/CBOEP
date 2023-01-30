# NIMF (Negative Interaction by Maximum Flow)

### Generation of negative enhancer-promoter interactions 

The main script is extended_region_research/main.py.

### Libraries 
---

| Library | Version |
| :---: | :---: |
| ```pandas``` | TD |
| ```numpy``` | TD |
| ```biopython``` | TD |
| ```gensim``` | TD |
| ```pybedtools``` | TD |
| ```sklearn``` | TD |
| `````` | TD |
| `````` | TD |
| `````` | TD |
| `````` | TD |
| `````` | TD |


### Directory structure
---

```
data_root_path 
├── bed  
│   ├── enhancer  
│   └── promoter  
├── fasta  
│   ├── enhancer  
│   └── promoter  
├── reference_genome  
├── d2v  
├── train  
├── result  
└── log  



extended_reginon_research  
├── main.py  
├── make_directory.py  
├── data_download.py  
├── data_processing.py  
├── my_doc2vec.py  
├── train_classifier.py  
├── t_sne.py  
├── make_args_logfile.py  
└── utils.py  
```

### Argument
---

| argument | default value | description |
| :---: | :---: | ---- |
| ```--data``` |||


