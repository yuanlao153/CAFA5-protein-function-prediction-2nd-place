"""
Standalone inference for trained LogReg (lin_t5_raw / lin_t5_cond) models.
Loads 5-fold model weights and produces oof_pred.pkl + test_pred.pkl.
"""
import argparse
import gc
import os
import sys

import cupy as cp
import joblib
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-m', '--model-name', type=str)
parser.add_argument('-d', '--device', type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from protlib.metric import obo_parser, Graph, get_topk_targets
    from protlib.models.prepocess import get_features_simple
    from protlib.models.logreg import LogRegMultilabel

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    model_config = config['base_models'][args.model_name]
    embeds_path = os.path.join(config['base_path'], config['embeds_path'])
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])
    models_path = os.path.join(config['base_path'], config['models_path'])

    # --- load ontology and get target columns ---
    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    split = [model_config['bp'], model_config['mf'], model_config['cc']]
    cols = []
    for n, i in enumerate(split):
        cols.extend(get_topk_targets(
            ontologies[n], i,
            train_path=os.path.join(config['base_path'], 'Train')
        ))

    # --- load train + test features ---
    model_dir = os.path.join(models_path, args.model_name)

    train_embeds = [os.path.join(embeds_path, x, 'train_embeds.npy') for x in model_config['embeds']]
    test_embeds = [os.path.join(embeds_path, x, 'test_embeds.npy') for x in model_config['embeds']]

    X_train, train_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/train_seq.feather'), train_embeds
    )
    X_test, test_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/test_seq.feather'), test_embeds
    )
    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")

    N_FOLDS = 5
    oof_pred = np.zeros((X_train.shape[0], len(cols)), dtype=np.float32)
    test_pred = np.zeros((X_test.shape[0], len(cols)), dtype=np.float32)

    for f in range(N_FOLDS):
        print(f"\n=== Fold {f} ===")
        model_path = os.path.join(model_dir, f'model_{f}.pkl')
        val_idx_path = os.path.join(model_dir, f'val_idx_f{f}.npy')

        if not os.path.exists(model_path):
            print(f"WARNING: {model_path} not found, skipping fold {f}")
            continue

        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        # OOF prediction using saved val indices
        if os.path.exists(val_idx_path):
            ts_idx = np.load(val_idx_path)
            print(f"Loaded val_idx from {val_idx_path} ({len(ts_idx)} samples)")
        else:
            print(f"WARNING: {val_idx_path} not found, skipping OOF for fold {f}")
            ts_idx = np.array([], dtype=int)

        if len(ts_idx) > 0:
            X_val = np.ascontiguousarray(X_train[ts_idx], dtype=np.float32)
            oof_fold = model.predict(X_val, batch_size=5000)
            oof_pred[ts_idx] += oof_fold
            print(f"OOF done, shape: {oof_fold.shape}")
            del X_val, oof_fold

        # test prediction
        X_test_cont = np.ascontiguousarray(X_test, dtype=np.float32)
        test_fold = model.predict(X_test_cont, batch_size=5000)
        del X_test_cont
        test_pred += test_fold
        print(f"Test done, shape: {test_fold.shape}")
        del test_fold

        del model
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    test_pred = test_pred / N_FOLDS

    joblib.dump(oof_pred, os.path.join(model_dir, 'oof_pred.pkl'))
    joblib.dump(test_pred, os.path.join(model_dir, 'test_pred.pkl'))
    print(f"\n>>> Done! Saved to {model_dir}")
    print(f"    oof_pred:  {oof_pred.shape}")
    print(f"    test_pred: {test_pred.shape}")
