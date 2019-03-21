import pandas as pd
import numpy as np

csv_path = "data/warfarin.csv"


def dose_to_action(dose):
    if dose < 21:
        return 0
    elif dose <= 49:
        return 1
    else:
        return 2


data = pd.read_csv(csv_path).dropna(axis=0, how='any', subset=['Age', 'Height (cm)', 'Weight (kg)', 'Therapeutic Dose of Warfarin'])

num_patient = data.iloc[:, 0].size

action = data['Therapeutic Dose of Warfarin'].apply(dose_to_action)

fixed_dose = np.ones(num_patient)

print("Fixed dose average return:", np.sum(fixed_dose == action) / num_patient)

WCDA = 4.0376 - \
       0.2546*data['Age'].str[0].astype(float) + \
       0.0118*data['Height (cm)'] + \
       0.0134*data['Weight (kg)'] - \
       0.6752*(data['Race'] == 'Asian').astype(float) + \
       0.4060*(data['Race'] == 'Black or African American').astype(float) + \
       0.0443*(data['Race'] == 'Unknown').astype(float) + \
       1.2799*(((data['Carbamazepine (Tegretol)'] == 1) |
               (data['Phenytoin (Dilantin)'] == 1)) |
               (data['Rifampin or Rifampicin'] == 1)).astype(float) - \
       0.5695*(data['Amiodarone (Cordarone)'] == 1).astype(float)

WCDA = (WCDA**2).apply(dose_to_action)
print("WCDA dose average return:", np.sum(WCDA == action) / num_patient)

