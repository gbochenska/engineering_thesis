import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

df = pd.read_csv('datasets/heart_2020_cleaned.csv')

print(df.mean())
print(df.median())
# X = df.drop(columns='HeartDisease')
# y = pd.DataFrame(df['HeartDisease'])

# #distribution of data in each class chart
# df.hist()
# plt.show()

# sns.countplot(x="HeartDisease", data=y)
# plt.title("Do you have a heart disease?")

# category = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
#             'PhysicalActivity', 'Asthma',
#             'KidneyDisease', 'SkinCancer', 'AgeCategory', 'Race',
#             'Diabetic', 'GenHealth', 'SleepTime']

# plt.subplots(figsize=(15,40))
# for i,column in  enumerate(category):
#     plt.subplot(len(category), 1, i+1)
#     plt.suptitle("Distribution of data in each class chart", fontsize=20, x=0.5, y=1)
#     sns.countplot(data=df, x=column)
#     plt.title(f"{column}")
#     plt.tight_layout()
#     plt.xticks([])
# plt.show()