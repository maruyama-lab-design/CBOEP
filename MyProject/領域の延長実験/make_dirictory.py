import os

def make_directory(args):
    os.system(f"mkdir -p {args.my_data_folder_path}/bed/enhancer")
    os.system(f"mkdir -p {args.my_data_folder_path}/bed/promoter")

    os.system(f"mkdir -p {args.my_data_folder_path}/fasta/enhancer")
    os.system(f"mkdir -p {args.my_data_folder_path}/fasta/promoter")

    os.system(f"mkdir -p {args.my_data_folder_path}/table/region/enhancer")
    os.system(f"mkdir -p {args.my_data_folder_path}/table/region/promoter")

    os.system(f"mkdir -p {args.my_data_folder_path}/model")

    os.system(f"mkdir -p {args.my_data_folder_path}/result")
    
    os.system(f"mkdir -p {args.my_data_folder_path}/train")
