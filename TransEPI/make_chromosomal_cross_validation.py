





commands = []
for train_cell in ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]:
    for test_cell in["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]:
        # if train_cell == "K562" or test_cell == "K562":
        if train_cell == "GM12878":
            command = f"python chromosomal_cross_validation.py --config chromosomal_cross_validation.json --train_cell {train_cell} --test_cell {test_cell}\n"
            commands.append(command)

with open(f"chromosomal_cross_validation.sh", "w") as f:
    f.writelines(commands)
