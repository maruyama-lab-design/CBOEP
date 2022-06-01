import argparse
import os
import datetime

import preprocess
import stage1
import stage2
import make_log

# これを実行していく


parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
parser.add_argument("--dataset", help="どのデータセットを使うか", default="new")
parser.add_argument("--cell_line", help="細胞株", default="K562")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--window", type=int, default=10)
parser.add_argument("--k_stride_set", help="k-merのk", default="1_1,2_2,3_3,4_4,5_5,6_6")
parser.add_argument("--embedding_vector_dimention", help="分散表現の次元", type=int, default=100)
parser.add_argument("--classifier", type=str, choices=["GBRT", "KNN", "SVM", "ALL"], default="GBRT", help="分類器に何を使うか")

parser.add_argument("--d2v_dir", type=str)
parser.add_argument("--seq_dir", type=str)
parser.add_argument("--train_dir", type=str)
parser.add_argument("--result_dir", type=str)
parser.add_argument("--reference_genome", type=str, default="/Users/ylwrvr/卒論/Koga_code/kmer-set/preprocess/hg19.fa")
args = parser.parse_args()



args.d2v_dir = os.path.join(os.path.dirname(__file__), "d2v")
args.seq_dir = os.path.join(os.path.dirname(__file__), "preprocess")
args.train_dir = os.path.join(os.path.dirname(__file__), "training_data")
args.reference_genome = os.path.join(os.path.dirname(__file__), "preprocess", "hg19.fa")


for cell_line in ["GM12878"]:
    # 時間計測__
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    d = now.strftime('%Y-%m-%d_%H:%M:%S')
    #__________

    #__________

    args.cell_line = cell_line
    args.result_dir = os.path.join(os.path.dirname(__file__), "result", d) # result dirは更新していく
    #__________

    os.system(f"mkdir -p {args.result_dir}")

    # preprocess.preprocess(args)
    # stage1.stage1(args)
    # stage2.stage2(args)
    make_log.make_log(args)
