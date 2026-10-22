# Import required libraries
import pandas as pd
from datasets import load_dataset
from IPython.display import display

# Created for investigation

# Load GLUE SST-2 dataset
dataset = load_dataset("nyu-mll/glue", "sst2")

# Display first 5 records from each dataset
print("Training Dataset")
display(dataset["train"].to_pandas().head(5))

print("Validation Dataset")
display(dataset["validation"].to_pandas().head(5))

print("Testing Dataset")
display(dataset["test"].to_pandas().head(5))
