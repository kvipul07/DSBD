import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print("Dataset:")
print(df.head())

plt.hist(df["fare"])

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")

plt.show()
