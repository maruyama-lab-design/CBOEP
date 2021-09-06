import os
import argparse

def data_download(args):

    for cell_line in args.cell_line_list:
        # enhancer
        os.system(f"wget {args.targetfinder_data_root_url}{cell_line}/output-ep/enhancers.bed -O {args.my_data_folder_path}bed/enhancer/{cell_line}_enhancers.bed")

        # promoter
        os.system(f"wget {args.targetfinder_data_root_url}{cell_line}/output-ep/promoters.bed -O {args.my_data_folder_path}bed/promoter/{cell_line}_promoters.bed")


    # reference genome
    os.system(f"wget {args.reference_genome_url} -O {args.my_data_folder_path}reference_genome/hg19.fa.gz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='targetfinderのgithubからenhancerとpromoterのbedfileをダウンロードするコードです.')
    parser.add_argument("-cell_line_list", nargs="+", help="細胞株の名前 (複数選択可能)", default=["GM12878"])
    parser.add_argument("--targetfinder_data_root_url", help="enhancer,promoterデータをダウンロードする際のtargetfinderのルートurl", default="https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/")
    parser.add_argument("--reference_genome_url", help="reference genome (hg19)をダウンロードする際のurl", default="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz")
    parser.add_argument("-my_data_folder_path", help="ダウンロードしたデータを置く場所を指定")
    args = parser.parse_args()

    data_download(args)
