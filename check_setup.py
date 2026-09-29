"""Verify the BERT environment: packages, device, and a quick model test.
Run from /Users/amit/BERT with bert-env activated:  python check_setup.py
"""
import sys
import time

print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")

try:
    import torch
    import transformers
    import datasets
    import accelerate
except ImportError as e:
    sys.exit(f"Missing package: {e.name}. Run: pip install torch transformers datasets evaluate accelerate")

print(f"torch       : {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"datasets    : {datasets.__version__}")
print(f"accelerate  : {accelerate.__version__}")

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"Device      : {device}")

from transformers import pipeline

print("\nLoading google-bert/bert-base-uncased (first run downloads ~440 MB)...")
fill = pipeline("fill-mask", model="google-bert/bert-base-uncased", device=device)

start = time.time()
results = fill("The capital of France is [MASK].")
elapsed = time.time() - start

print("\nPredictions for: 'The capital of France is [MASK].'")
for r in results:
    print(f"  {r['token_str']:>10}  {r['score']:.3f}")
print(f"\nInference time: {elapsed:.2f}s on {device}")

ok = results[0]["token_str"] == "paris"
print("\nSETUP OK ✅" if ok else "\nModel ran, but top prediction wasn't 'paris' — check output above.")