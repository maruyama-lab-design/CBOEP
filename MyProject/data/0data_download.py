import os

def func(args):

    cell_line_list = ["GM12878"]

    targetfinder_data_root_path = "https://github.com/shwhalen/targetfinder/raw/master/paper/targetfinder/"
    my_data_folder_path =  "/Users/ylwrvr/卒論/Koga_code/MyProject/data/"

    for cell_line in cell_line_list:
        # enhancer
        os.system(f"wget {targetfinder_data_root_path}{cell_line}/output-ep/enhancers.bed -O {my_data_folder_path}bed/enhancer/{cell_line}_enhancers.bed")

        # promoter
        os.system(f"wget {targetfinder_data_root_path}{cell_line}/output-ep/promoters.bed -O {my_data_folder_path}bed/promoter/{cell_line}_promoters.bed")


    # reference genome
    os.system(f"wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz -O {my_data_folder_path}reference_genome/hg19.fa.gz")

if __name_ == '__main__':
    parser = argparse.ArgumentParser(description='This program makes ...')
    parser.add_argument("cell_line", help="...") # the first argument. 
    parser.add_argument("targetfinder_data_root_path", help="......")
    # my_data_folder_path =
    args = parser.parse_args()