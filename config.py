
"""
config.py — Central configuration for the entire pipeline.

All scripts (data.py, train.py, eval.py) import from here.
Override device via the DEVICE environment variable (e.g. DEVICE=cpu or DEVICE=cuda).
"""

import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = 'distilbert-base-cased'
CACHED_MODEL_DIR = 'distilbert-reviews-genres'
DEVICE_NAME = os.environ.get('DEVICE', 'mps')  # 'mps' | 'cuda' | 'cpu'

# ── Tokenizer ──────────────────────────────────────────────────────────────────
MAX_LENGTH = 512

# ── Data ───────────────────────────────────────────────────────────────────────
HEAD = 10000           # max reviews to read per genre before sampling
SAMPLE_SIZE = 2000     # reviews sampled per genre
TRAIN_PER_GENRE = 800  # reviews per genre used for training (remainder → test)

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

# ── Training ───────────────────────────────────────────────────────────────────
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
    report_to=[],           # set to 'wandb' in Task 4
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_REPORT_PATH = 'eval_report.json'
