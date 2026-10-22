import argparse
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "google-bert/bert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init",
        choices=["pretrained", "random"],
        required=True,
        help="Use pretrained BERT weights or random weights."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional small dataset for testing code."
    )

    parser.add_argument(
        "--freeze_layers",
        type=int,
        choices=[6, 12],
        required=True,
        help="Freeze the first 6 Transformer layers or the entire BERT encoder."
    )

    args = parser.parse_args()

    print("\n==============================")
    print("BERT SST-2 Experiment")
    print("==============================")
    print("Initialization:", args.init)

    # ------------------------------------------------
    # Load SST-2
    # ------------------------------------------------

    dataset = load_dataset("nyu-mll/glue", "sst2")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=128
        )

    tokenized = dataset.map(
        tokenize,
        batched=True
    )

    train_dataset = tokenized["train"]
    validation_dataset = tokenized["validation"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )

    # ------------------------------------------------
    # Create model
    # ------------------------------------------------

    if args.init == "pretrained":

        print("Loading PRETRAINED BERT weights")

        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

    else:

        print("Creating BERT with RANDOM weights")

        config = BertConfig.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

        model = BertForSequenceClassification(config)

    # ------------------------------------------------
    # Freeze selected BERT layers
    # ------------------------------------------------

    if args.freeze_layers == 6:
        # Freeze Transformer layers 0 through 5 only.
        # Embeddings, pooler, layers 6-11 and classifier remain trainable.
        for layer in model.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

    elif args.freeze_layers == 12:
        # Freeze the entire BERT encoder, including embeddings and pooler.
        # Only the sentiment classification head remains trainable.
        for param in model.bert.parameters():
            param.requires_grad = False

    print(f"Frozen BERT layers: {args.freeze_layers}")

    # ------------------------------------------------
    # Parameter count
    # ------------------------------------------------

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # ------------------------------------------------
    # Training setup
    # ------------------------------------------------

    output_directory = f"outputs/{args.init}_freeze{args.freeze_layers}"

    training_args = TrainingArguments(

        output_dir=output_directory,

        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.batch_size,

        per_device_eval_batch_size=args.batch_size,

        num_train_epochs=args.epochs,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="accuracy",

        seed=42,

        report_to="none"
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

    start_time = time.time()

    trainer.train()

    training_time = time.time() - start_time

    # ------------------------------------------------
    # Evaluate
    # ------------------------------------------------

    results = trainer.evaluate()

    print("\n==============================")
    print("RESULTS")
    print("==============================")

    print("Initialization:", args.init)

    print(
        f"Validation Accuracy: "
        f"{results['eval_accuracy']:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time / 60:.2f} minutes"
    )

    print(
        f"Trainable Parameters: "
        f"{trainable_parameters:,}"
    )

    trainer.save_model(
        f"{output_directory}/final_model"
    )


if __name__ == "__main__":
    main()

