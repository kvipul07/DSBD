import pandas as pd

df = pd.read_csv("file_name.csv")

print("Dataset:")
print(df.head())

print("\nSummary Statistics:")
print(df.groupby("species").mean())

print("\nMedian:")
print(df.groupby("species").median())

print("\nMinimum Values:")
print(df.groupby("species").min())

print("\nMaximum Values:")
print(df.groupby("species").max())

print("\nStandard Deviation:")
print(df.groupby("species").std())
