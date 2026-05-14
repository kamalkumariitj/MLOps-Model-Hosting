"""
eval.py — Evaluation, metrics, and saving results.

Reads:
  - test_dataset.pickle
  - test_labels.pickle
  - label_maps.pickle
  - distilbert-reviews-genres/   (fine-tuned model)

Outputs:
  - eval_report.json   (full classification report)
"""

import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import json
import pickle

from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report

from utils import compute_metrics

# ── Configuration ──────────────────────────────────────────────────────────────
CACHED_MODEL_DIR = 'distilbert-reviews-genres'
DEVICE_NAME = 'mps'        # change to 'cuda' or 'cpu' as needed
EVAL_REPORT_PATH = 'eval_report.json'
# ───────────────────────────────────────────────────────────────────────────────


def main():
    # Load test data and label maps
    test_dataset = pickle.load(open('test_dataset.pickle', 'rb'))
    test_labels = pickle.load(open('test_labels.pickle', 'rb'))
    label_maps = pickle.load(open('label_maps.pickle', 'rb'))
    id2label = label_maps['id2label']

    print(f'Test size: {len(test_dataset)} | Labels: {list(id2label.values())}')

    # Load the fine-tuned model
    model = DistilBertForSequenceClassification.from_pretrained(CACHED_MODEL_DIR).to(DEVICE_NAME)

    # Trainer is needed only to call evaluate/predict; no training args required
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Built-in evaluation (loss + accuracy + F1) ─────────────────────────────
    print('\nRunning evaluation…')
    eval_results = trainer.evaluate()
    print(eval_results)

    # ── Detailed predictions and classification report ──────────────────────────
    predicted_results = trainer.predict(test_dataset)
    predicted_ids = predicted_results.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]

    print('\nClassification Report:')
    print(classification_report(test_labels, predicted_labels))

    # ── Save classification report to JSON ─────────────────────────────────────
    report = classification_report(
        test_labels,
        predicted_labels,
        target_names=list(id2label.values()),
        output_dict=True,
    )
    # Attach top-level eval metrics
    report['eval_loss'] = eval_results.get('eval_loss')
    report['eval_accuracy'] = eval_results.get('eval_accuracy')
    report['eval_f1'] = eval_results.get('eval_f1')

    with open(EVAL_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\nSaved evaluation report to {EVAL_REPORT_PATH}')


if __name__ == '__main__':
    main()
