#__________________
# よく使うファイルのパス名や変数を記録する
# argparse の勉強中
# まだ実用段階ではない
#__________________

import argparse    # 1. argparseをインポート

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
#------例------
# parser.add_argument('arg1', help='この引数の説明（なくてもよい）')    # 必須の引数を追加
# parser.add_argument('arg2', help='foooo')
# parser.add_argument('--arg3')    # オプション引数（指定しなくても良い引数）を追加
# parser.add_argument('-a', '--arg4')   # よく使う引数なら省略形があると使う時に便利
parser.add_argument("--targetfinder_root_path", default="/Users/ylwrvr/卒論/Koga_code/targetfinder")
parser.add_argument("--data_root_path", default="/Users/ylwrvr/卒論/Koga_code/MyProject/data")
parser.add_argument("--reference_genome_path", default="/Users/ylwrvr/卒論/Koga_code/MyProject/data/reference_genome/hg19.fa")
parser.add_argument("--cell_line", nargs="+", default=["GM12878", "K562"])
parser.add_argument("--neighbor_length", default=5000)

args = parser.parse_args()    # 4. 引数を解析

print('targetfinder path = '+args.targetfinder_root_path)
print('data path = '+args.data_root_path)
print('cell line =', args.cell_line)
