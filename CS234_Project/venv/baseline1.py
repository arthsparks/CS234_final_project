import pandas as pd

csv_path = "/Users/shengji/Dropbox/CS234_Project/data/warfarin.csv"

data = pd.read_csv(csv_path, header=0)

true_dose = data.loc[:, 'Therapeutic Dose of Warfarin']


fix_acc = true_dose.between(3.*7,7.*7, inclusive=True).astype(float).sum()/true_dose.size

print(fix_acc)

if __name__ == '__main__':
    print("Hello World")
