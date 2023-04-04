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
