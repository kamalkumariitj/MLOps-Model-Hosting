"""
run.py — Orchestrator: runs the full pipeline in order.

Usage:
  python run.py                  # full pipeline (data → train → eval)
  python run.py --skip-data      # skip data download/encoding (reuse pickles)
  python run.py --skip-train     # skip training (reuse saved model)
  python run.py --only-eval      # only run evaluation
"""

import argparse
import sys
import time


def run_stage(name, fn):
    print(f'\n{"=" * 60}')
    print(f'  STAGE: {name}')
    print(f'{"=" * 60}')
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f'\n  ✓ {name} completed in {elapsed:.1f}s')


def main():
    parser = argparse.ArgumentParser(description='MLOps Assignment 2 — Full Pipeline')
    parser.add_argument('--skip-data',  action='store_true', help='Skip data download/encoding (reuse existing pickles)')
    parser.add_argument('--skip-train', action='store_true', help='Skip training (reuse existing saved model)')
    parser.add_argument('--only-eval',  action='store_true', help='Run evaluation stage only')
    args = parser.parse_args()

    if args.only_eval:
        args.skip_data = True
        args.skip_train = True

    # ── Stage 1: Data ──────────────────────────────────────────────────────────
    if not args.skip_data:
        import data
        run_stage('Data Loading & Encoding', data.main)
    else:
        print('\n[Skipping] Data stage — using existing pickle files.')

    # ── Stage 2: Training ──────────────────────────────────────────────────────
    if not args.skip_train:
        import train
        run_stage('Model Training', train.main)
    else:
        print('[Skipping] Training stage — using existing saved model.')

    # ── Stage 3: Evaluation ────────────────────────────────────────────────────
    import eval as evaluation
    run_stage('Evaluation', evaluation.main)

    print(f'\n{"=" * 60}')
    print('  Pipeline complete.')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
