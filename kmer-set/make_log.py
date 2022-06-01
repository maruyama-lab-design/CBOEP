import pandas as pd
import csv
import os

def make_log(args):
    dic = vars(args)

    with open(os.path.join(args.result_dir, "log.csv"), 'w') as f:  
        writer = csv.writer(f)
        for k, v in dic.items():
            writer.writerow([k, v])

