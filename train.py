"""
train.py — Model loading, Trainer setup, and training loop.

Reads:
  - train_dataset.pickle
  - test_dataset.pickle
  - label_maps.pickle

Outputs:
  - distilbert-reviews-genres/   (saved fine-tuned model)
  - results/                     (checkpoints)
"""

import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import pickle

from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer

from utils import compute_metrics

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = 'distilbert-base-cased'
DEVICE_NAME = 'mps'        # change to 'cuda' (GPU) or 'cpu' as needed
CACHED_MODEL_DIR = 'distilbert-reviews-genres'

TRAINING_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='steps',
    save_strategy='steps',
    save_steps=500,
    load_best_model_at_end=True,
    report_to=[],           # W&B integration added in Task 4
)
# ───────────────────────────────────────────────────────────────────────────────


def main():
    # Load datasets and label maps
    train_dataset = pickle.load(open('train_dataset.pickle', 'rb'))
    test_dataset = pickle.load(open('test_dataset.pickle', 'rb'))
    label_maps = pickle.load(open('label_maps.pickle', 'rb'))
    id2label = label_maps['id2label']

    print(f'Train size: {len(train_dataset)} | Test size: {len(test_dataset)}')
    print(f'Number of labels: {len(id2label)}')

    # Load pre-trained DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
    ).to(DEVICE_NAME)

    training_args = TrainingArguments(**TRAINING_ARGS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print('Starting training…')
    trainer.train()

    print(f'Saving model to {CACHED_MODEL_DIR}/')
    trainer.save_model(CACHED_MODEL_DIR)
    print('Training complete.')


if __name__ == '__main__':
    main()
