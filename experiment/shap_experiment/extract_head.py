# extract first 100 lines in csv file and save to csv file
csv_file_path = "/home/two/LMTDE/data/datasets/nacc_new/naccImg_validation_normed.csv.backup"
target_csv_path = "/home/two/LMTDE/data/datasets/nacc_new/naccImg_validation_normed.csv"
with open(csv_file_path, 'r') as csv_file:
    with open(target_csv_path, 'w') as target_csv_file:
        for i in range(7900):
            line = csv_file.readline()
            target_csv_file.write(line)
