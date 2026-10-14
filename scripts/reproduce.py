#!/usr/bin/env python3
"""Execute the final notebook and save metrics with input/environment provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root)
    parser.add_argument('--output-dir', type=Path, default=root / 'results')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Maximum execution time per notebook cell, in seconds')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    missing = [name for name in ('train.csv', 'test.csv') if not (data / name).is_file()]
    if missing:
        parser.error('Original competition data required: ' + ', '.join(missing))
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(root / 'damage_prediction.ipynb', as_version=4)
    export = '''import json, platform, importlib.metadata
from pathlib import Path
tn, fp, fn, tp = confusion_matrix(y_val, preds_pp, labels=[0, 1]).ravel()
report = {
    "evaluation": "2015 tuning set; thresholds and blend selected using its labels",
    "validation_rows": int(len(y_val)),
    "seed": SEED, "folds": N_FOLDS,
    "balanced_accuracy": float(balanced_accuracy_score(y_val, preds_pp)),
    "accuracy": float(accuracy_score(y_val, preds_pp)),
    "roc_auc": float(roc_auc_score(y_val, p_val_final)),
    "damage_recall": float(tp / (tp + fn)),
    "no_damage_specificity": float(tn / (tn + fp)),
    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    "global_threshold": float(t_use),
    "phase_thresholds": {str(k): float(v) for k, v in phase_thr.items()},
    "python": platform.python_version(),
    "packages": {p: importlib.metadata.version(p) for p in
        ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm"]}
}
Path(EXPORT_PATH).write_text(json.dumps(report, indent=2) + "\\n")
print(json.dumps(report, indent=2))
'''
    export = 'EXPORT_PATH = ' + repr(str(output / 'metrics.json')) + '\n' + export
    notebook.cells.append(nbformat.v4.new_code_cell(export))
    try:
        ExecutePreprocessor(timeout=args.timeout, kernel_name='python3').preprocess(
            notebook, {'metadata': {'path': str(data)}})
    finally:
        # Partial outputs also help diagnose a failed run; they aren't a success score.
        nbformat.write(notebook, output / 'damage_prediction.executed.ipynb')
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    provenance = {'sha256': {name: sha(data / name) for name in ('train.csv', 'test.csv')},
                  'notebook_sha256': sha(root / 'damage_prediction.ipynb')}
    (output / 'inputs.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Completed evaluation saved to {output}')
    print(f'Test predictions written to {data / "submission.csv"}')


if __name__ == '__main__':
    main()
