import pandas as pd

df = pd.read_csv("spambase.csv", header=None)
df.columns = [f"feature_{i}" for i in range(57)] + ["class"]


print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget value counts:")
print(df["class"].value_counts())
