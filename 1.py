import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("iris.csv")

print("Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

le = LabelEncoder()

df["species"] = le.fit_transform(df["species"])

print("\nUpdated Dataset:")
print(df.head())
