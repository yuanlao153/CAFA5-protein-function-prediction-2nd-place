# CAFA5 Protein Function Prediction - 完整运行命令

> 所有命令在项目根目录 `/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place` 执行
> RAPIDS_ENV = `rapids-pb-env/bin/python`
> PYTORCH_ENV = `pytorch-env/bin/python`
> CONFIG_PATH = `config.yaml`

---

## 1. 数据准备

### 1.1 解析 FASTA 文件
```bash
./rapids-pb-env/bin/python protlib/scripts/parse_fasta.py --config-path config.yaml
```

### 1.2 生成 real_targets + prior + nulls（传播开 + 去重 + 二值化）
> `protlib/scripts/create_helpers.py` 已修改：
> - 添加 `--ontology` 按本体并行
> - 添加 `drop_duplicates` 去重
> - 添加 `trg[trg > 0] = 1.0` 二值化
> - `--propagate True` 开启 GO DAG 传播

```bash
# 不并行（一次性跑完三个本体）
./rapids-pb-env/bin/python protlib/scripts/create_helpers.py \
    --config-path config.yaml \
    --batch-size 10000 \
    --propagate True

# 或者三路并行（更快）
./rapids-pb-env/bin/python protlib/scripts/create_helpers.py --config-path config.yaml --batch-size 10000 --propagate True --ontology biological_process >> log/create_helpers_bp.log 2>&1 &
./rapids-pb-env/bin/python protlib/scripts/create_helpers.py --config-path config.yaml --batch-size 10000 --propagate True --ontology molecular_function >> log/create_helpers_mf.log 2>&1 &
./rapids-pb-env/bin/python protlib/scripts/create_helpers.py --config-path config.yaml --batch-size 10000 --propagate True --ontology cellular_component >> log/create_helpers_cc.log 2>&1 &
```

#### 为什么 `--propagate True` 效果更好

**核心原因：让训练标签符合 GO 的 True Path Rule（DAG 向上闭包），保证 `real_targets` 里的 0/1/NaN 语义正确。**

##### Step 1：传播前的原始矩阵

`create_helpers.py` 第 105-106 行，从 `train_terms.tsv` 构建 `trg` 矩阵（protein × term）：

```python
trg = np.zeros((num.shape[0], ont.idxs), dtype=np.float32)
np.add.at(trg, (trm_ont['n'].values, trm_ont['id'].values), 1)
```

此时矩阵只有被显式标注的 term 为 1，其余为 0。但 `train_terms.tsv` 通常只标到叶子节点：

```
蛋白 P001: "ATP binding" (GO:0043531) = 1
            "binding" (GO:0005488)     = 0  ← 应该是 1，但没标注
            "molecular_function"        = 0  ← 应该是 1，但没标注
```

##### Step 2：传播（`propagate_target`，第 44-54 行）

```python
def propagate_target(mat, G):
    for f in G.order:                       # 按拓扑序：叶子 → 根
        adj = G.terms_list[f]['children']   # term f 的子节点列表
        if len(adj) == 0:
            continue                         # 没有子节点，跳过
        prop_max_cpu(mat, f, np.asarray(adj))
```

`G.order` 是从叶到根的拓扑排序，保证处理一个 term 时其所有子节点已经处理过。

`prop_max_cpu`（第 31-41 行）是 Numba JIT 加速的 CPU 函数：

```python
@njit
def prop_max_cpu(mat, k, adj):
    for i in prange(mat.shape[0]):     # 并行遍历所有蛋白
        if mat[i, k] == 1:
            continue                    # 已经是 1，跳过
        for j in adj:                   # 遍历子节点
            if mat[i, j] == 1:          # 任一子节点是 1
                mat[i, k] = 1           # 父节点也设为 1
                continue
```

等价语义：**parent = max(children)**，即只要有一个子节点为正，父节点就是正。

```
传播后: ATP_binding=1 → binding=1 → molecular_function=1  ✅ DAG 闭包
```

##### Step 3：NaN 生成依赖传播结果（第 115-122 行）

```python
for k, node in enumerate(ont.terms_list):
    adj = node['adj']                   # k 的父节点列表
    if len(adj) > 0:
        na = np.nonzero(np.nansum(trg[:, adj], axis=1) == 0)[0]
        trg[na, k] = np.nan
```

逻辑：如果某个蛋白在一个 term 的**所有父节点上都是 0**，则该 term 对该蛋白不可判定 → 置 NaN。

这个规则的前提是父节点已经正确传播——否则大量本应是 1 的父节点还是 0，导致本应可判定的 term 被错误置为 NaN。

##### 传播 vs 不传播，对真实数据的影响

以分子功能（MF）本体为例：

| | propagate=False | propagate=True |
|------|:--:|:--:|
| 父节点正例数 | 偏少（人工标注只到叶子） | 补全（DAG 闭包） |
| NaN 比例（nulls） | 偏高（父节点全 0 → term 变 NaN） | 正常 |
| prior_cnd（条件先验） | 偏低（很多本应是 1 的被标为 0/NaN） | 准确 |
| 可训练样本数 | 减少 | 增加 |

##### 对 downstream 模型的影响链条

```
propagate=True
  → real_targets/*.parquet: 0/1 更准确, NaN 更少
  → prior.pkl / nulls.pkl: 统计值正确
  → raw GBDT: 父节点不再被错误当负例学
  → cond GBDT: NaN 率降低 → 更多 term 有有效训练信号
  → GCN stacking: prior_raw / prior_cnd 值域正确 → 模型融合校准准确
  → 最终 Fmax 提升
```

> 如果不传播（`default=False`），大量上层 term 的 0 实际是"没标注"而非"不存在"，模型会学到系统性错误负例，且 nulls 虚高导致 cond 模型训练不充分。

### 1.3 下载外部数据（详见 notebook 1.4 节）
```bash
./rapids-pb-env/bin/python protlib/scripts/downloads/dw_goant.py --config-path config.yaml
./rapids-pb-env/bin/python protlib/scripts/parse_go_single.py --file goa_uniprot_all.gaf.216.gz --config-path config.yaml
./rapids-pb-env/bin/python protlib/scripts/parse_go_single.py --file goa_uniprot_all.gaf.214.gz --config-path config.yaml --output old214
./rapids-pb-env/bin/python protlib/scripts/prop_tsv.py --path {file} --graph Train/go-basic.obo --output {name} --device 0 --batch_size 30000 --batch_inner 5000
./rapids-pb-env/bin/python protlib/scripts/reproduce_mt.py --path temporal --graph Train/go-basic.obo
```

### 1.4 NN 准备
```bash
pytorch-env/bin/python nn_solution/prepare.py --config-path config.yaml
```

---

## 2. 嵌入推理

```bash
pytorch-env/bin/python nn_solution/t5.py --config-path config.yaml --device 0
pytorch-env/bin/python nn_solution/esm2sm.py --config-path config.yaml --device 0
```

---

## 3. Base 模型训练

### 3.1 py-boost GBDT（4500 输出：BP 3000 + MF 1000 + CC 500）

```bash
# pb_t54500_cond (T5, conditional)
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_pb.py --config-path config.yaml --model-name pb_t54500_cond --device 0 >> log/train_pb_t54500_cond.log 2>&1 &

# pb_t5esm4500_raw (T5+ESM, raw)
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_pb.py --config-path config.yaml --model-name pb_t5esm4500_raw --device 1 >> log/train_pb_t5esm4500_raw.log 2>&1 &

# pb_t5esm4500_cond (T5+ESM, conditional)
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_pb.py --config-path config.yaml --model-name pb_t5esm4500_cond --device 2 >> log/train_pb_t5esm4500_cond.log 2>&1 &

# pb_t54500_raw (T5, raw)
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_pb.py --config-path config.yaml --model-name pb_t54500_raw --device 3 >> log/train_pb_t54500_raw.log 2>&1 &
```

### 3.2 py-boost IF 模型（T5 + struct）

```bash
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_pb.py --config-path config.yaml --model-name pb_t5if4500_raw --device 2 >> log/train_pb_t5if4500_raw.log 2>&1 &
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_pb.py --config-path config.yaml --model-name pb_t5if4500_cond --device 3 >> log/train_pb_t5if4500_cond.log 2>&1 &
```

### 3.3 LogReg（13500 输出：BP 10000 + MF 2000 + CC 1500）
> `protlib/scripts/train_lin.py` 已修改：添加 `--folds` 参数分折并行

```bash
# lin_t5_raw 分 3 路并行
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_lin.py --config-path config.yaml --model-name lin_t5_raw --device 0 --folds 0,1 >> log/train_lin_t5_raw_f01.log 2>&1 &
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_lin.py --config-path config.yaml --model-name lin_t5_raw --device 1 --folds 2,3 >> log/train_lin_t5_raw_f23.log 2>&1 &
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_lin.py --config-path config.yaml --model-name lin_t5_raw --device 2 --folds 4 >> log/train_lin_t5_raw_f4.log 2>&1 &

# 合并
./rapids-pb-env/bin/python protlib/scripts/merge_folds.py --config-path config.yaml --model-name lin_t5_raw

# lin_t5_cond
nohup ./rapids-pb-env/bin/python -u protlib/scripts/train_lin.py --config-path config.yaml --model-name lin_t5_cond --device 3 >> log/train_lin_t5_cond.log 2>&1 &
```

### 3.4 NN（预训练权重）
```bash
pytorch-env/bin/python nn_solution/train_models.py --config-path config.yaml --device 0
pytorch-env/bin/python nn_solution/inference_models.py --config-path config.yaml --device 0
pytorch-env/bin/python nn_solution/make_pkl.py --config-path config.yaml
```

---

## 4. GCN 训练

### 4.1 原始 GCN（5 模型，hidden=16）

```bash
nohup pytorch-env/bin/python -u protnn/scripts/train_gcn.py --config-path config.yaml --ontology cc --device 1 >> log/train_gcn_cc.log 2>&1 &
nohup pytorch-env/bin/python -u protnn/scripts/train_gcn.py --config-path config.yaml --ontology mf --device 2 >> log/train_gcn_mf.log 2>&1 &
# BP 中断后从 checkpoint 恢复：
nohup pytorch-env/bin/python -u test_addiif/train_gcn_resume.py --config-path config.yaml --ontology bp --device 3 >> log/train_gcn_bp_resume.log 2>&1 &
```

### 4.2 GCN IF 版本（7 模型，+pb_t5if4500_raw/cond，hidden=16）
> 新增 `test_addiif/train_gcn_if.py`：`models_config` +2 IF 模型，`GCNStacker(7,1,...)`

```bash
nohup pytorch-env/bin/python -u test_addiif/train_gcn_if.py --config-path config.yaml --ontology cc --device 0 >> log/train_gcn_if_cc.log 2>&1 &
nohup pytorch-env/bin/python -u test_addiif/train_gcn_if.py --config-path config.yaml --ontology mf --device 3 >> log/train_gcn_if_mf.log 2>&1 &
nohup pytorch-env/bin/python -u test_addiif/train_gcn_if.py --config-path config.yaml --ontology bp --device 0 --epochs 10 >> log/train_gcn_if_bp_10ep.log 2>&1 &
```

---

## 5. GCN 推理
> `protnn/scripts/predict_gcn.py` 已修改：TTA 间内存清理 + num_workers=8
> **需要 124GB 内存**

```bash
nohup pytorch-env/bin/python -u protnn/scripts/predict_gcn.py --config-path config.yaml --device 0 >> log/predict_gcn_v6.log 2>&1 &
```

输出：
```
models/gcn/pred_tta_0.tsv
models/gcn/pred_tta_1.tsv
models/gcn/pred_tta_2.tsv
models/gcn/pred_tta_3.tsv
```

---

## 6. 后处理 → 最终提交

```bash
# 聚合 4 路 TTA
./rapids-pb-env/bin/python protlib/scripts/postproc/collect_ttas.py --config-path config.yaml --device 0

# min propagate
./rapids-pb-env/bin/python protlib/scripts/postproc/step.py --config-path config.yaml --device 0 --batch_size 30000 --batch_inner 3000 --lr 0.7 --direction min

# max propagate
./rapids-pb-env/bin/python protlib/scripts/postproc/step.py --config-path config.yaml --device 0 --batch_size 30000 --batch_inner 3000 --lr 0.7 --direction max

# 合并生成最终提交
./rapids-pb-env/bin/python protlib/scripts/postproc/make_submission.py --config-path config.yaml --device 0 --max-rate 0.5
```

最终输出：`sub/submission.tsv`

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | 1-4 张 32GB+ VRAM |
| 内存 | 训练 62GB+，推理 124GB+ |
| 磁盘 | 30GB+ (overlay) |
| Python | rapids-pb-env / pytorch-env / rapids-gcn-env |
| 嵌入 | T5 + ESM + struct (Foldseek) |

## 修改清单

| 文件 | 改动 |
|------|------|
| `protlib/scripts/create_helpers.py` | +`--ontology` 并行, +去重, +二值化 |
| `protlib/scripts/train_lin.py` | +`--folds` 分折, +显存清理 |
| `protlib/scripts/merge_folds.py` | 新文件，合并分折结果 |
| `protnn/scripts/predict_gcn.py` | TTA 间 `del` + `gc.collect()`, num_workers 调整 |
| `test_addiif/train_gcn_if.py` | IF 版 GCN, 7 模型, 输出到 `model_if/` |
| `test_addiif/train_gcn_if_h32.py` | IF 版 GCN, 支持 `--hidden-size`, `--output-name` |
| `test_addiif/train_gcn_resume.py` | GCN 断点续训，从 SWA checkpoint 恢复 |

## 各阶段运行时间（4×RTX 4080 Super 32GB 实测）

### 数据准备
| 步骤 | 时间 |
|------|------|
| parse_fasta | ~2 分钟 |
| create_helpers (三路并行) | ~30 分钟 |
| 外部数据下载+解析 | ~10 小时（含 FTP 下载） |
| NN prepare | ~5 分钟 |

### 嵌入推理
| 嵌入 | 时间 |
|------|------|
| T5 | ~2 小时 |
| ESM | ~1 小时 |
| struct | ~0.5 小时 |

### Base 模型（5 fold CV，单卡）
| 模型 | 时间 | 输出大小 |
|------|------|------|
| pb_t54500_cond | ~6 小时 | 2.6G (oof) + 3.8G (test) |
| pb_t54500_raw | ~4 小时 | 同上 |
| pb_t5esm4500_cond | ~6.5 小时 | 同上 |
| pb_t5esm4500_raw | ~4.5 小时 | 同上 |
| pb_t5if4500_cond | ~7 小时 | 同上 |
| pb_t5if4500_raw | ~5 小时 | 同上 |
| lin_t5_raw (三路并行) | ~1.7 小时 | 7.4G (oof) + 12G (test) |
| lin_t5_cond | ~2.5 小时 | 同上 |
| nn_serg | ~1 小时 × 12 次 | 2.2G |

### GCN 训练（单卡）
| 本体 | 原始 (5模型) | IF (7模型) |
|------|------|------|
| CC | ~2 小时 | ~2.5 小时 |
| MF | ~5 小时 | ~6 小时 |
| BP | ~13 小时 | 未完成 |

### GCN 推理（单卡，4 路 TTA）
| 阶段 | 时间 |
|------|------|
| 预测 (BP+MF+CC × 4 TTA) | ~2.5 小时 |
| 后处理 (collect + step + make_submission) | ~10 分钟 |

### 全程总耗时（4 卡并行）
| 阶段 | 时间 |
|------|------|
| 准备 + 嵌入 | ~3 小时 |
| Base 模型（4 卡轮流） | ~18 小时 |
| GCN 训练（3 卡并行） | ~13 小时 |
| 推理 + 后处理 | ~3 小时 |
| **总计** | **~37 小时** |
