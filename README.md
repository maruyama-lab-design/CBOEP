# Koga_code

## TargetFinderにあるエンハンサー・プロモーターの領域を延長し，各配列をdoc2vecによる分散表現によって相互作用を予測するモデル

extended_region_researchディレクトリ内のmain.pyを実行してください．

###引数###

```-my_data_folder_path```
必要なデータがこのフォルダーをルートパスとして生成されます．必ず指定してください

```--make_directory```
必要なディレクトリを```-my_data_folder_path```中に生成します．

```--download_reference_genome```
hg19のリファレンスゲノムをgenome browserからダウンロードします．

```--cell_line_list```
実験に用いる細胞株を指定してください．デフォルトは```K562```

```--E_extended_left_length```
エンハンサーの上流をどれだけ伸ばすか．デフォルトは10

```--E_extended_right_length```
エンハンサーの下流をどれだけ伸ばすか．デフォルトは20

```--P_extended_left_length```
プロモーターの上流をどれだけ伸ばすか．デフォルトは30

```--P_extended_right_length```
プロモーターの下流をどれだけ伸ばすか． デフォルトは40

```--embedding_vector_dimention```
paragraph vector の次元．デフォルトは100です．

```--way_of_kmer```
k-merの切り方("normal" or "random")
固定長なら"normal"，ランダム長なら"random"を指定してください．デフォルトは"normal"

```--k```
```way_of_kmer```がnormalの場合のk-merのk．デフォルトは6

```--stride```
```way_of_kmer```がnormalの場合のstride．デフォルトは1

```--sentence_cnt```
```way_of_kmer```がrandomの場合，一本のsequenceから複製されるのsentence個数

```--k_min```
```way_of_kmer```がrandomの場合のk_min

```--k_max```
```way_of_kmer```がrandomの場合のk_max

###実行例###  
```main.py -my_data_folder_path /Users/ylwrvr/卒論/Koga_code/MyProject/data --cell_line_list GM12878 --make_directory --download_reference_genome --share_doc2vec ```
