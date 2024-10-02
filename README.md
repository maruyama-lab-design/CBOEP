# Generators of Negative enhancer-promoter pairs, CBMF (Class Balanced negative set by Maximum-Flow) and CBGS (Class Balanced negative set by Gibbs Sampling)

This repository contains files related to our methods, CBMF and CBGS, which generate a set of negative enhancer-promoter interactions (EPIs) from a set of positive EPIs. 
The implementations of the methods are `cbmf.py` and `cbgs.py` in the top directory. 

The scripts we coded in this study [1] has been published in the preceedings of ACM-BCB 2023.

# Datasets
CBMF and CBGS require a positive EPIs set as input to generate a negative dataset. 

The input files for these methods, pre-processed from the original BENGI [2] positive sets and 
the positive sets used in the work of TargetFinder [3], are stored in the directory
[input_to_neg_generator](https://github.com/maruyama-lab-design/CBOEP/tree/main/input_to_neg_generator).

The negative sets generated using the above input sets 
are stored in the directory
[output_from_neg_generator](https://github.com/maruyama-lab-design/CBOEP/tree/main/output_from_neg_generator). 

If you want to generate negative EPI sets from a positive EPI set,
please note that 
the positive set should be in the csv format with the following columns:

| Column | Description |
| :---: | --- |
| ```label``` | Numeric ```1``` for positive EPI, ```0``` for negative EPI |
| ```enhancer_distance_to_promoter``` | Distance between the enhancer and the promoter |
| ```enhancer_chrom``` | Chromosome number of the enhancer |
| ```enhancer_start``` | Start position of the enhancer |
| ```enhancer_end``` | End position of the enhancer |
| ```enhancer_name``` | Name of the enhancer, such as `GM12878\|chr16:88874-88924` |
| ```promoter_chrom``` | Chromosome number of the promoter |
| ```promoter_start``` | Start position of the promoter |
| ```promoter_end``` | End position of the promoter |
| ```promoter_name``` | Name of the promoter, such as `GM12878\|chr16:103009-103010`|

If you want to use an EPI set to an EPI predictor in directory
[EPI_predictor](https://github.com/maruyama-lab-design/CBOEP/tree/main/EPI_predictor),
the reference genome version shold be GRCh37/hg19.

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
| ```-i``` ||Path to an input EPI dataset file.|
| ```-o``` ||Path to an output EPI dataset file.|
| ```--dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--concat``` |False|Whether or not to concatenate the CBMF negative set with the positive set given as input. If not given, only the CBMF negative set will be output.|



## CBMF Execution example
```  
python cbmf.py \
-i ./input_to_neg_generator/normalized_BENGI/GM12878.csv \
-o ./output_from_neg_generator/BENGI-P_CBMF-N/GM12878.csv \
--dmax 2500000 \
--dmin 0 \
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
| ```-i``` ||Path to an input EPI dataset file.|
| ```-o``` ||Path to an output EPI dataset file.|
| ```--dmax``` |2,500,000|Upper bound of enhancer-promoter distance for newly generated negative EPIs.|
| ```--dmin``` |0|Lower bound of enhancer-promoter distance for newly generated negative EPIs.|
|```--T```|40,000|Number of sampling iteration|
| ```--concat``` ||If given, the CBGS negative set is concatenated with the positive set given as input. If not given, only the CBGS negative set will be output.|
|```--make_fig```||If given, a figure which shows plots of the mean of positive/negative class imbalance of all enhancers and promoters for each sampling iteration is made.|
|```--out_figfile```||If ```--make_fig``` is given, a figure is saved in this path.|


## CBGS Execution example
```  
python cbgs.py \
-i ./input_to_neg_generator/normalized_BENGI/GM12878.csv \
-o ./output_from_neg_generator/BENGI-P_CBGS-N/GM12878.csv \
--dmax 2500000 \
--dmin 0 \
--concat \
--make_fig \
--out_figfile ./output_from_neg_generator/BENGI-P_CBGS-N/GM12878.png
```


# Reference
[1]
Tsukasa K, Osamu M.
CBOEP: Generating negative enhancer-promoter interactions to train classifiers.
In Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB ’23).
2023;Article 27:1–6.

[2]
Moore JE, Pratt HE, Purcaro MJ, et al.
A curated benchmark of enhancer-gene interactions for evaluating enhancer-target gene prediction methods.
Genome Biol.
2020;21(1):17.

[3]
Whalen S, Truty RM, Pollard KS.
Enhancer-promoter interactions are encoded by complex genomic signatures on looping chromatin.
Nat Genet.
2016;48(5):488–496. 





