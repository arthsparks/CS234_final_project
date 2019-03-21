import pandas as pd


def dose_to_action(dose):
    if dose < 21:
        return 0
    elif dose <= 49:
        return 1
    else:
        return 2


def gender_to_index(gender):
    if gender == "male":
        return 1
    elif gender == 'female':
        return 2
    else:
        return 3


def race_to_index(race):
    if race == "Asian":
        return 1
    elif race == 'Black or African American':
        return 2
    elif race == 'White':
        return 3
    else:
        return 4

def ethninity_to_index(ethninity):
    if ethninity == "Hispanic or Latino":
        return 1
    elif ethninity == 'not Hispanic or Latino':
        return 2
    else:
        return 3

def true_false_to_index(boolean):
    if boolean:
        return 1


def generate_data(save_txt=True, num_labels=3):
    # File Path
    csv_path = "data/warfarin.csv"
    txt_path = "data/processed.txt"

    # Parameters
    num_bins_height = 10
    num_bins_weight = 10

    # Load and clean raw data
    data = pd.read_csv(csv_path).dropna(axis=0, how='any', subset=['Therapeutic Dose of Warfarin'])

    if num_labels == 3:
        data_out = pd.Series.to_frame(data['Therapeutic Dose of Warfarin'].apply(dose_to_action))
    else:
        data_out = pd.Series.to_frame(pd.cut(data['Therapeutic Dose of Warfarin'], num_labels, labels=False))

    # Add age
    age = data['Age'].str[0].str.get_dummies()
    data_out = pd.concat([data_out, age], axis=1)

    # Add height
    height = pd.cut(data['Height (cm)'], num_bins_height, labels=False, include_lowest=True).astype(str).str.get_dummies()
    data_out = pd.concat([data_out, height], axis=1)

    # Add weight
    weight = pd.cut(data['Weight (kg)'], num_bins_weight, labels=False, include_lowest=True).astype(str).str.get_dummies()
    data_out = pd.concat([data_out, weight], axis=1)

    # Add gender
    gender = data['Gender'].apply(gender_to_index).astype(str).str.get_dummies()
    data_out = pd.concat([data_out, gender], axis=1)

    # Add race
    race = data['Race'].apply(race_to_index).astype(str).str.get_dummies()
    data_out = pd.concat([data_out, race], axis=1)

    # Add ethinity
    ethninity = data['Ethnicity'].apply(ethninity_to_index).astype(str).str.get_dummies()
    data_out = pd.concat([data_out, ethninity], axis=1)

    # Add indication
    indication = data['Indication for Warfarin Treatment'].astype(str).str[0].str.get_dummies()
    data_out = pd.concat([data_out, indication], axis=1)

    # Add miscellaneous
    col_index_to_add = [9, 10, 11, *range(13, 32), 33, *range(36, 63)]
    for i in col_index_to_add:
        data_out = pd.concat([data_out, data.iloc[:, i].astype(str).str.get_dummies()], axis=1)

    # Random shuffle sample
    data_out.sample(frac=1)

    # Write to txt
    if save_txt:
        total_points = data_out.iloc[:, 0].size
        num_features = len(data_out.columns) - 1

        with open(txt_path, "w") as f:
            f.write("{} {} {}\n".format(total_points, num_features, num_labels))
            for index, row in data_out.iterrows():
                col_index = 0
                for col, val in row.iteritems():
                    if col == 'Therapeutic Dose of Warfarin':
                        f.write("{0:.0f}".format(row['Therapeutic Dose of Warfarin']))
                    elif val:
                        f.write(" {:.0f}:{:.6f}".format(col_index, val))
                    col_index += 1
                f.write("\n")

    return data_out, num_labels

if __name__ == '__main__':
    generate_data()

