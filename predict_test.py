import csv
import time

from datasets import load_dataset
from transformers import pipeline

# ------------------------------------------------
# 1. Load fine-tuned BERT
# ------------------------------------------------

MODEL_PATH = "outputs/pretrained/final_model"


print("Loading fine-tuned BERT model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer="google-bert/bert-base-uncased",
    device=0
)

# ------------------------------------------------
# 2. Load official SST-2 test dataset
# ------------------------------------------------

dataset = load_dataset("nyu-mll/glue", "sst2")
test_dataset = dataset["test"]

print("Test samples:", len(test_dataset))

# ------------------------------------------------
# 3. Predict sentiment
# ------------------------------------------------

start_time = time.time()

# Convert dataset column to a Python list
sentences = list(test_dataset["sentence"])

predictions = classifier(
    sentences,
    truncation=True,
    max_length=128,
    batch_size=32
)

prediction_time = time.time() - start_time

# ------------------------------------------------
# 4. Display first 10 predictions
# ------------------------------------------------

for i in range(10):
    label = predictions[i]["label"]
    score = predictions[i]["score"]

    sentiment = "Positive" if label == "LABEL_1" else "Negative"

    print("\nSentence:", test_dataset[i]["sentence"])
    print("Prediction:", sentiment)
    print(f"Confidence: {score:.4f}")

# ------------------------------------------------
# 5. Save test predictions
# ------------------------------------------------

with open("SST-2.tsv", "w", newline="") as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["index", "prediction"])

    for example, result in zip(test_dataset, predictions):
        label = int(result["label"].split("_")[-1])
        writer.writerow([example["idx"], label])

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print("Total predictions:", len(predictions))
print(f"Inference time: {prediction_time:.2f} seconds")
print("Predictions saved to SST-2.tsv")

