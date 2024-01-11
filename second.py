import numpy as np
import pandas as pd
from pathlib import Path

xpt_file = "C:/Users/gboch/Desktop/inzynierka/engineering_thesis/second/datasets/LLCP2022.XPT"
# df = pd.read_sas(xpt_file, encoding='utf-8')
# df.to_csv('C:/Users/gboch/Desktop/inzynierka/engineering_thesis/second/datasets/new.csv', index=False)
df = pd.read_csv('datasets/new.csv')
print(df.head())

NEW_VAR_NAMES = [
    "HeartDisease", 
    "BMI", 
    "Smoking", 
    "AlcoholDrinking", 
    "Stroke",
    "PhysicalHealth", 
    "MentalHealth",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "SleepTime",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
    "State"
]

var_list = [
    "_MICHD", 
    "_BMI5",
    "_SMOKER3",  
    "DRNKANY6",
    "CVDSTRK3", 
    "PHYSHLTH", 
    "MENTHLTH",
    "DIFFWALK",
    "_SEX", 
    "_AGEG5YR",
    "_RACEGR4",
    "DIABETE4",
    "EXERANY2",
    "GENHLTH", 
    "SLEPTIM1",
    "ASTHMA3",
    "CHCKDNY2", 
    "CHCSCNC1",  
    "_STATE"]

heart_df = df[var_list]
heart_df.columns = NEW_VAR_NAMES

STATE = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
    66: "Guam",
    72: "Puerto Rico",
    78: "Virgin Islands"
}

SEX = {1: 'Male', 2: 'Female'}

GEN_HEALTH = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}

YES_NO_QUESTIONS = {1: 'Yes', 2: 'No'}

SLEEP_TIME = lambda x: np.where(x > 24, np.nan, x)
PHYS_MEN_HEALTH = {77: np.nan,
               88: 0,
               99: np.nan
                  }

AGE_CATEGORY = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80 or older"
}

SMOKER_STATUS = {
    1: "Yes",
    2: "Yes",
    3: "No",
    4: "No"
}

RACE = {
    1: "White",
    2: "Black",
    3: "Other",
    4: "Multiracial",
    5: "Hispanic"
}

heart_copy_df = heart_df.copy()
heart_copy_df['State'] = heart_copy_df['State'].map(STATE)
heart_copy_df['Sex'] = heart_copy_df['Sex'].map(SEX)
heart_copy_df['GenHealth'] = heart_copy_df['GenHealth'].map(GEN_HEALTH)
heart_copy_df['PhysicalHealth'] = heart_copy_df['PhysicalHealth'].replace(PHYS_MEN_HEALTH)
heart_copy_df['MentalHealth'] = heart_copy_df['MentalHealth'].replace(PHYS_MEN_HEALTH)
heart_copy_df['PhysicalActivity'] = heart_copy_df['PhysicalActivity'].map(YES_NO_QUESTIONS)
heart_copy_df['SleepTime'] = heart_copy_df['SleepTime'].apply(SLEEP_TIME)
heart_copy_df['HeartDisease'] = heart_copy_df['HeartDisease'].map(YES_NO_QUESTIONS)
heart_copy_df['Stroke'] = heart_copy_df['Stroke'].map(YES_NO_QUESTIONS)
heart_copy_df['Asthma'] = heart_copy_df['Asthma'].map(YES_NO_QUESTIONS)
heart_copy_df['SkinCancer'] = heart_copy_df['SkinCancer'].map(YES_NO_QUESTIONS)
heart_copy_df['KidneyDisease'] = heart_copy_df['KidneyDisease'].map(YES_NO_QUESTIONS)
heart_copy_df['DiffWalking'] = heart_copy_df['DiffWalking'].map(YES_NO_QUESTIONS)
heart_copy_df['Smoking'] = heart_copy_df['Smoking'].map(SMOKER_STATUS)
heart_copy_df['Race'] = heart_copy_df['Race'].map(RACE)
heart_copy_df['AgeCategory'] = heart_copy_df['AgeCategory'].map(AGE_CATEGORY)
heart_copy_df['BMI'] = heart_copy_df['BMI'] / 100
heart_copy_df['AlcoholDrinking'] = heart_copy_df['AlcoholDrinking'].map(YES_NO_QUESTIONS)

print(heart_copy_df.head())

heart_copy_df.to_csv('C:/Users/gboch/Desktop/inzynierka/engineering_thesis/second/datasets/new.csv', index=False)