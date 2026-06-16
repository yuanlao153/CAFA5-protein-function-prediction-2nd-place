# import argparse
# import os
# import sys
# import cupy as cp
# import joblib
# import numpy as np
# import yaml

# sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config-path', type=str)
# parser.add_argument('-m', '--model-name', type=str)

# # parser.add_argument('-t', '--target', type=str)
# # parser.add_argument('-hp', '--helpers', type=str)

# # parser.add_argument('-tre', '--train-embeds', type=str, nargs='+')
# # parser.add_argument('-tse', '--test-embeds', type=str, nargs='+')
# #
# # parser.add_argument('-b', '--bp', type=int, default=0)
# # parser.add_argument('-m', '--mf', type=int, default=0)
# # parser.add_argument('-c', '--cc', type=int, default=0)

# # parser.add_argument('-cnd', '--conditional', type=str)
# # parser.add_argument('-o', '--output', type=str)
# parser.add_argument('-d', '--device', type=str)
# # parser.add_argument('-g', '--graph', type=str)

# if __name__ == '__main__':

#     args = parser.parse_args()

#     # Optional: set the device to run
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

#     try:
#         from protlib.metric import obo_parser, Graph, get_topk_targets
#         from protlib.models.prepocess import get_features_simple, get_targets_from_parquet
#         from protlib.models.logreg import LogRegMultilabel

#     except ImportError:
#         print('Alarm')
#         pass

#     from py_boost import GradientBoosting
#     from py_boost.multioutput.sketching import RandomProjectionSketch

#     with open(args.config_path) as f:
#         config = yaml.safe_load(f)

#     model_config = config['base_models'][args.model_name]
#     graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
#     embeds_path = os.path.join(config['base_path'], config['embeds_path'])
#     helpers_path = os.path.join(config['base_path'], config['helpers_path'])

#     ontologies = []
#     for ns, terms_dict in obo_parser(graph_path).items():
#         ontologies.append(Graph(ns, terms_dict, None, True))

#     split = [model_config['bp'], model_config['mf'], model_config['cc']]
#     cols = []

#     for n, i in enumerate(split):
#         cols.extend(get_topk_targets(
#             ontologies[n],
#             i,
#             train_path=os.path.join(config['base_path'], 'Train')
#         ))

#     fillna = not model_config['conditional']  # args.conditional == 'false'
#     print(fillna)
#     Y = get_targets_from_parquet(
#         os.path.join(helpers_path, 'real_targets'),
#         ontologies,
#         split,
#         ids=cols,
#         fillna=fillna
#     )

#     train_embeds = [os.path.join(embeds_path, x, 'train_embeds.npy') for x in model_config['embeds']]
#     test_embeds = [os.path.join(embeds_path, x, 'test_embeds.npy') for x in model_config['embeds']]

#     X, train_idx = get_features_simple(
#         os.path.join(helpers_path, 'fasta/train_seq.feather'), train_embeds
#     )

#     X_test, test_idx = get_features_simple(
#         os.path.join(helpers_path, 'fasta/test_seq.feather'), test_embeds
#     )

#     print(X.shape, X_test.shape)

#     N_FOLDS = 5
#     # assume embedding sum is our key
#     key = np.array(list(map(hash, X.sum(axis=1))))
#     test_key = np.array(list(map(hash, X_test.sum(axis=1))))

#     np.random.seed(42)
#     folds = np.unique(key)
#     np.random.shuffle(folds)
#     folds = np.array_split(folds, N_FOLDS)

#     oof_pred = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
#     test_pred = np.zeros((X_test.shape[0], len(cols)), dtype=np.float32)

#     output = os.path.join(config['base_path'], config['models_path'], args.model_name)
#     os.makedirs(output, exist_ok=True)

#     for f in range(N_FOLDS):
#         # get indexers
#         test_sl = np.isin(key, folds[f])

#         pred_idx = np.arange(X_test.shape[0])
#         tr_idx, ts_idx = np.nonzero(~test_sl)[0], np.nonzero(test_sl)[0]
#         print(tr_idx.shape, ts_idx.shape)

#         # train model
#         model = LogRegMultilabel(alpha=0.00001)
#         model.fit(X[tr_idx], Y[tr_idx])
#         joblib.dump(model, os.path.join(output, f'model_{f}.pkl'))

#         # oof prediction
#         oof_pred[ts_idx] += model.predict(X[ts_idx])
#         # test prediction
#         test_pred += model.predict(X_test)


#     test_pred = test_pred / N_FOLDS

#     joblib.dump(oof_pred, os.path.join(output, 'oof_pred.pkl'))
#     joblib.dump(test_pred, os.path.join(output, 'test_pred.pkl'))
import argparse
import os
import sys
import cupy as cp
import joblib
import numpy as np
import yaml
import gc

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-m', '--model-name', type=str)
parser.add_argument('-d', '--device', type=str)
parser.add_argument('--folds', type=str, default=None,
                    help='comma-separated fold indices to run, e.g. "0,1". default: all 5 folds')

if __name__ == '__main__':

    args = parser.parse_args()

    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print(f"--- Starting Linear Model: {args.model_name} ---")

    try:
        from protlib.metric import obo_parser, Graph, get_topk_targets
        from protlib.models.prepocess import get_features_simple, get_targets_from_parquet
        from protlib.models.logreg import LogRegMultilabel

    except ImportError:
        print('Alarm: Import failed.')
        pass

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    # 检查配置名是否正确
    if args.model_name not in config['base_models']:
        print(f"❌ Config Error: {args.model_name} not found in base_models!")
        sys.exit(1)

    model_config = config['base_models'][args.model_name]
    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    embeds_path = os.path.join(config['base_path'], config['embeds_path'])
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    split = [model_config['bp'], model_config['mf'], model_config['cc']]
    cols = []

    for n, i in enumerate(split):
        cols.extend(get_topk_targets(
            ontologies[n],
            i,
            train_path=os.path.join(config['base_path'], 'Train')
        ))

    fillna = not model_config['conditional']
    print(f"Fillna strategy: {fillna}")
    
    Y = get_targets_from_parquet(
        os.path.join(helpers_path, 'real_targets'),
        ontologies,
        split,
        ids=cols,
        fillna=fillna
    )

    train_embeds = [os.path.join(embeds_path, x, 'train_embeds.npy') for x in model_config['embeds']]
    test_embeds = [os.path.join(embeds_path, x, 'test_embeds.npy') for x in model_config['embeds']]

    X, train_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/train_seq.feather'), train_embeds
    )

    X_test, test_idx = get_features_simple(
        os.path.join(helpers_path, 'fasta/test_seq.feather'), test_embeds
    )

    print(f"Data Shapes - Train: {X.shape}, Test: {X_test.shape}")

    N_FOLDS = 5
    key = np.array(list(map(hash, X.sum(axis=1))))
    
    np.random.seed(42)
    folds = np.unique(key)
    np.random.shuffle(folds)
    folds = np.array_split(folds, N_FOLDS)

    # parse which folds to run
    if args.folds is not None:
        fold_list = [int(x) for x in args.folds.split(',')]
        print(f"Running folds: {fold_list}")
    else:
        fold_list = list(range(N_FOLDS))

    oof_pred = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
    test_pred = np.zeros((X_test.shape[0], len(cols)), dtype=np.float32)

    output = os.path.join(config['base_path'], config['models_path'], args.model_name)
    os.makedirs(output, exist_ok=True)

    for f in fold_list:
        print(f"\n=== Fold {f} / 4 ===")
        test_sl = np.isin(key, folds[f])
        tr_idx, ts_idx = np.nonzero(~test_sl)[0], np.nonzero(test_sl)[0]

        # 1. 显式转换数据格式 (防止 GPU 报错)
        X_train = np.ascontiguousarray(X[tr_idx], dtype=np.float32)
        Y_train = np.ascontiguousarray(Y[tr_idx], dtype=np.float32)

        # 2. 训练模型
        print("Training LogReg (alpha=0.0000001)...")
        model = LogRegMultilabel(alpha=0.0000001)

        try:
            model.fit(X_train, Y_train)
            print("Convergence reached.")
        except Exception as e:
            print(f"❌ Training Error: {e}")
            sys.exit(1)

        # 3. 保存模型
        joblib.dump(model, os.path.join(output, f'model_{f}.pkl'))

        # 4. 预测
        print("Predicting OOF...")
        X_val = np.ascontiguousarray(X[ts_idx], dtype=np.float32)
        oof_fold = model.predict(X_val, batch_size=5000)
        del X_val
        # save per-fold oof + test + val indices for merge
        np.save(os.path.join(output, f'val_idx_f{f}.npy'), ts_idx)
        joblib.dump(oof_fold, os.path.join(output, f'oof_pred_f{f}.pkl'))
        oof_pred[ts_idx] += oof_fold
        del oof_fold

        print("Predicting Test...")
        X_test_cont = np.ascontiguousarray(X_test, dtype=np.float32)
        test_fold = model.predict(X_test_cont, batch_size=5000)
        del X_test_cont
        # save per-fold test
        joblib.dump(test_fold, os.path.join(output, f'test_pred_f{f}.pkl'))
        test_pred += test_fold
        del test_fold

        # 5. 清理显存
        print("Cleaning GPU memory...")
        del model, X_train, Y_train
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        gc.collect()

    # 全五折时输出 merge 结果
    if len(fold_list) == N_FOLDS:
        test_pred = test_pred / N_FOLDS
        joblib.dump(oof_pred, os.path.join(output, 'oof_pred.pkl'))
        joblib.dump(test_pred, os.path.join(output, 'test_pred.pkl'))
        print(">>> Done! Merged oof_pred.pkl + test_pred.pkl saved.")
    else:
        print(f">>> Partial folds {fold_list} done. Per-fold files saved.")
        print("    Run all 5 folds, then combine with merge_folds.py")