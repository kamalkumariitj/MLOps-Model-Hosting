"""
data.py — Data loading, sampling, train/test split, and encoding.

Outputs written to disk:
  - genre_reviews_dict.pickle   (raw reviews per genre)
  - train_dataset.pickle        (MyDataset, encoded training set)
  - test_dataset.pickle         (MyDataset, encoded test set)
  - label_maps.pickle           (dict with 'label2id' and 'id2label')
  - test_labels.pickle          (list of raw string labels for the test set)
"""

import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import gzip
import json
import pickle
import random

import requests
from transformers import DistilBertTokenizerFast

from utils import MyDataset, build_label_maps

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = 'distilbert-base-cased'
MAX_LENGTH = 512
HEAD = 10000        # max reviews to read per genre before sampling
SAMPLE_SIZE = 2000  # reviews sampled per genre
TRAIN_PER_GENRE = 800
# ───────────────────────────────────────────────────────────────────────────────

GENRE_URL_DICT = {
    'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


def load_reviews(url, head=HEAD, sample_size=SAMPLE_SIZE):
    """Stream reviews from a gzipped JSON URL and return a random sample."""
    reviews = []
    response = requests.get(url, stream=True)
    response.raise_for_status()
    print(f'  HTTP {response.status_code}')
    with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            reviews.append(d['review_text'])
            if head is not None and i + 1 >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_reviews(cache_path='genre_reviews_dict.pickle'):
    """Load raw reviews for all genres, using a local cache if available."""
    if os.path.exists(cache_path):
        print(f'Loading cached reviews from {cache_path}')
        return pickle.load(open(cache_path, 'rb'))

    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f'Loading reviews for genre: {genre}')
        genre_reviews_dict[genre] = load_reviews(url)

    pickle.dump(genre_reviews_dict, open(cache_path, 'wb'))
    print(f'Saved reviews to {cache_path}')
    return genre_reviews_dict


def split_data(genre_reviews_dict, train_per_genre=TRAIN_PER_GENRE, total_per_genre=1000):
    """Split reviews into train and test lists."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(total_per_genre, len(reviews)))
        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def encode_data(train_texts, train_labels, test_texts, test_labels, model_name=MODEL_NAME, max_length=MAX_LENGTH):
    """Tokenize texts and encode labels; return MyDataset objects and label maps."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    label2id, id2label = build_label_maps(train_labels)

    print('Encoding training texts…')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    print('Encoding test texts…')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    return train_dataset, test_dataset, label2id, id2label


def main():
    genre_reviews_dict = load_all_reviews()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)

    print(f'Train: {len(train_texts)} reviews | Test: {len(test_texts)} reviews')

    train_dataset, test_dataset, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    pickle.dump(train_dataset, open('train_dataset.pickle', 'wb'))
    pickle.dump(test_dataset, open('test_dataset.pickle', 'wb'))
    pickle.dump({'label2id': label2id, 'id2label': id2label}, open('label_maps.pickle', 'wb'))
    pickle.dump(test_labels, open('test_labels.pickle', 'wb'))

    print('Saved: train_dataset.pickle, test_dataset.pickle, label_maps.pickle, test_labels.pickle')


if __name__ == '__main__':
    main()
