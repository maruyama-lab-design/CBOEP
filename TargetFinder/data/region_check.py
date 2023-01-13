import pandas as pd

dataname_list = ["BENGI", "TargetFinder"]
datatype_list = ["original", "maxflow_2500000", "maxflow_5000000", "maxflow_10000000", "maxflow_9999999999"]
cell_line_list = ["GM12878"]

for dataname in dataname_list:
    for datatype in datatype_list:
        for cell_line in cell_line_list:
            print(dataname, datatype, cell_line)
            filename = f"/Users/ylwrvr/卒論/Koga_code/TargetFinder/data/epw/wo_feature/{dataname}/{datatype}/{cell_line}.csv"
            df = pd.read_csv(filename)

            for i, pair in df.iterrows():
                for region in ["enhancer", "promoter", "window"]:
                    if pair[f"{region}_start"] >= pair[f"{region}_end"]:
                        print(i, region, pair[f"{region}_start"], pair[f"{region}_end"])