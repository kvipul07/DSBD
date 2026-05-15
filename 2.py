import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("file_name.csv")

print("Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nUpdated Missing Values:")
print(df.isnull().sum())

plt.boxplot(df["Marks"])
plt.title("Boxplot")
plt.show()

Q1 = df["Marks"].quantile(0.25)
Q3 = df["Marks"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Marks"] >= lower) & (df["Marks"] <= upper)]

df["Marks"] = np.log(df["Marks"])

print("\nUpdated Dataset:")
print(df.head())
