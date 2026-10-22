import pandas as pd

# Load predictions from both models
a = pd.read_csv("SST-2.tsv", sep="\t")
b = pd.read_csv("SST-2-6.tsv", sep="\t")

# Match predictions using the test record index
df = a.merge(b, on="index", suffixes=("_A", "_B"), validate="one_to_one")

# Check that both files contain the same test records
assert len(df) == len(a) == len(b), "Test indices do not match"

# Compare predictions
different = df[df["prediction_A"] != df["prediction_B"]]

print("Total test records:", len(df))
print("Same predictions:", len(df) - len(different))
print("Different predictions:", len(different))
print("Difference percentage:", round(len(different) / len(df) * 100, 2), "%")

print("\nFirst 10 different predictions:")
print(different.head(10).to_string(index=False))
