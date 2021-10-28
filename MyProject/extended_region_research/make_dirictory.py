import os

def make_directory(args):
    os.system(f"mkdir -p {args.my_data_folder_path}/bed/enhancer")
    os.system(f"mkdir -p {args.my_data_folder_path}/bed/promoter")
    # os.system(f"mkdir -p {args.my_data_folder_path}/bed/tss")

    os.system(f"mkdir -p {args.my_data_folder_path}/fasta/enhancer")
    os.system(f"mkdir -p {args.my_data_folder_path}/fasta/promoter")

    os.system(f"mkdir -p {args.my_data_folder_path}/table/region/enhancer")
    os.system(f"mkdir -p {args.my_data_folder_path}/table/region/promoter")

    os.system(f"mkdir -p {args.my_data_folder_path}/reference_genome")

    os.system(f"mkdir -p {args.my_data_folder_path}/figure")

    os.system(f"mkdir -p {args.my_data_folder_path}/d2v")

    os.system(f"mkdir -p {args.my_data_folder_path}/result")

    os.system(f"mkdir -p {args.my_data_folder_path}/train")

    os.system(f"mkdir -p {args.my_data_folder_path}/log")
