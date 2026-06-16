import argparse
import os
import sys
import glob
import joblib
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-m', '--model-name', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    output = os.path.join(config['base_path'], config['models_path'], args.model_name)

    # find all per-fold files
    fold_files = sorted(glob.glob(os.path.join(output, 'oof_pred_f*.pkl')))
    if not fold_files:
        print(f"No per-fold files found in {output}")
        sys.exit(1)

    fold_indices = [int(f.split('_f')[-1].split('.')[0]) for f in fold_files]
    print(f"Found folds: {fold_indices}")

    # determine shapes from first fold
    sample = joblib.load(os.path.join(output, f'oof_pred_f{fold_indices[0]}.pkl'))
    test_sample = joblib.load(os.path.join(output, f'test_pred_f{fold_indices[0]}.pkl'))
    n_train = None

    # try to infer full train size from val indices
    for fi in fold_indices:
        val_idx = np.load(os.path.join(output, f'val_idx_f{fi}.npy'))
        if n_train is None or val_idx.max() >= n_train:
            n_train = val_idx.max() + 1

    oof_pred = np.zeros((n_train, sample.shape[1]), dtype=np.float32)
    test_pred = np.zeros_like(test_sample)

    for fi in fold_indices:
        oof_fold = joblib.load(os.path.join(output, f'oof_pred_f{fi}.pkl'))
        test_fold = joblib.load(os.path.join(output, f'test_pred_f{fi}.pkl'))
        val_idx = np.load(os.path.join(output, f'val_idx_f{fi}.npy'))

        oof_pred[val_idx] += oof_fold
        test_pred += test_fold
        print(f"  Fold {fi}: oof={oof_fold.shape}, test={test_fold.shape}, val_idx={len(val_idx)}")

    N_FOLDS = 5
    test_pred = test_pred / N_FOLDS

    joblib.dump(oof_pred, os.path.join(output, 'oof_pred.pkl'))
    joblib.dump(test_pred, os.path.join(output, 'test_pred.pkl'))
    print(f">>> Merged {len(fold_indices)}/{N_FOLDS} folds → oof_pred.pkl + test_pred.pkl saved to {output}")
