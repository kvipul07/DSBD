# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("file_name.csv")

# print("Dataset:")
# print(df.head())

# print("\nData Types:")
# print(df.dtypes)

# df.hist(figsize=(10, 8))

# plt.show()

# df.plot(kind="box", subplots=True, layout=(2, 2), figsize=(10, 8))

# plt.show()

☺
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['species'] = iris.target

# Map target values to species names
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# ----------------------------------------
# 1. List down features and their types
# ----------------------------------------

print("\nFeatures and Data Types:\n")
print(df.dtypes)

print("\nDataset Information:\n")
print(df.info())

# ----------------------------------------
# 2. Histogram for each feature
# ----------------------------------------

df.hist(figsize=(12, 8), bins=15)
plt.suptitle("Histograms of Iris Dataset Features")
plt.show()

# ----------------------------------------
# 3. Boxplot for each feature
# ----------------------------------------

plt.figure(figsize=(12, 8))

for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()

# ----------------------------------------
# 4. Compare distributions and identify outliers
# ----------------------------------------

print("\nInference:")
print("1. Sepal length and sepal width are approximately normally distributed.")
print("2. Petal length and petal width show clear separation among species.")
print("3. Sepal width contains some outliers visible in the boxplot.")
print("4. Petal features have wider variation compared to sepal features.")
print("5. Iris-setosa species generally has smaller petal dimensions.")