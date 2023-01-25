import pickle
import gzip
import os

all_chrom = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
chr2seq = {}

for chrom in all_chrom:
    with gzip.open(os.path.join(os.path.dirname(__file__), "raw_seq", f"{chrom}.fa.gz"), mode='rt') as f:
        data = f.read()

        chrom  = data.split('\n')[0][1:]
        print(chrom)

        seq_list = data.split('\n')[1:]
        seq = "".join(seq_list).lower()
        print(len(seq))

        chr2seq[chrom] = seq

with open(os.path.join(os.path.dirname(__file__), "raw_seq", "chrom2seq.pkl"), "wb") as tf:
    pickle.dump(chr2seq, tf)