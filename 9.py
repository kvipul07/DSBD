import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

print("Dataset:")
print(df.head())

sns.boxplot(x="sex", y="age", hue="survived", data=df)

plt.title("Age Distribution")

plt.show()
