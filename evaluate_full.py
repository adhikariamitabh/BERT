
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Change this path to evaluate a different experiment
MODEL_PATH = "outputs/pretrained/final_model"
# evaluate_freeze6.py

# Load saved model
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer="google-bert/bert-base-uncased",
    device=0
)

# Load validation data
validation = load_dataset("nyu-mll/glue", "sst2")["validation"]

sentences = list(validation["sentence"])
actual = list(validation["label"])

# Generate predictions
results = classifier(
    sentences,
    truncation=True,
    max_length=128,
    batch_size=32
)

predicted = [
    int(result["label"].split("_")[-1])
    for result in results
]

# Calculate metrics
print("\nVALIDATION RESULTS")
print("Model:", MODEL_PATH)
print("Samples:", len(sentences))
print("Accuracy:", round(accuracy_score(actual, predicted), 4))

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    target_names=["Negative", "Positive"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))

