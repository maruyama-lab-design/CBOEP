import datetime

def make_args_logfile(args):   
    dt_now = datetime.datetime.now()
    with open(f"{args.my_data_folder_path}/log/{dt_now.strftime('%m-%d_%H:%M:%S')}", "w") as logfile:
        for arg, value in sorted(vars(args).items()):
            logfile.write(f"{arg}: {value}\n")
