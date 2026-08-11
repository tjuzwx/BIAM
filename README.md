# BIAM：面向特征缺失的概率双层交互可加模型

本仓库实现大论文第四章概率双层交互可加模型（Probabilistic Bilevel Interactive Additive Model，BIAM）。

## 1. 模型介绍

BIAM 的线性预测量由四部分组成：

\[
\widehat\eta_i=\beta_0+F_{\mathrm{obs},i}+F_{\mathrm{miss},i}+F_{\mathrm{int},i}.
\]

- 观测主效应：训练集中位数仅用于闭合输入，数值效应乘以观测指示，缺失值不会作为真实观测参与响应。
- 缺失主效应：使用中心化缺失指示 \(m_{ij}-\bar m_j\)。
- 缺失交互效应：只构造有向项 \(m_{ij}o_{ik}q_{jk}(x_{ik})\)，并按训练集支持度 \(n_{jk}^{MO}\ge n_{\min}\) 筛选候选交互。
- Hinge 表示：每个结点同时包含正向和反向 Hinge 基；结点、中心化常数、均值、标准差和中位数全部只由训练集估计。
- 离散结构：每个完整效应组使用 Bernoulli 门控，训练时采样离散结构，部署时按 \(\pi_g\ge0.5\) 得到硬结构。
- 样本加权：权重网络以逐样本损失为输入、Sigmoid 为输出；下层将当前权重作为损失梯度系数，不引入权重网络对损失输入的额外下层导数。

优化器实现以下论文步骤：

1. 使用偶数个反变量 Bernoulli 结构样本；默认 \(S=8\)。
2. 每轮开始冻结惰性结构缓存的逻辑快照，所有结构从同一轮首状态评价。
3. 每个结构执行 \(K_{in}-1\) 次普通加权下层更新，再执行一次保留权重网络计算图的虚拟更新；默认 \(K_{in}=5\)。
4. 结构参数使用排除当前样本及其反变量伙伴的留一对基线和分数函数梯度，并加入期望组数与 Bernoulli KL 正则。
5. 权重网络使用所有结构共享元风险的一步截断元梯度。
6. 两个上层模块同步更新后，从同一个 \(\bar\theta\) 使用更新后的权重网络计算实际下层步；同一轮重复结构的结果先平均再写回缓存。
7. 调参集只用于超参数和早停。训练结束后固定硬结构和权重网络，在训练集与元数据集的并集上重拟合；测试集只在最终模型冻结后评价。

核心实现位于：

- `models/biam_additive_model.py`：Hinge 基、三类效应、经验中心化、候选交互与硬结构；
- `models/biam_weighting_network.py`：损失到样本权重的映射；
- `gradients/biam_optimizer.py`：反变量结构梯度、截断元梯度、惰性缓存、早停和最终重拟合；
- `data/biam_data_generator.py`：论文仿真公式、四路划分、缺失机制、标签/响应异常与真实回归外层交叉验证；
- `biam_main.py`：多随机种子实验和完整复现记录。

## 2. 大论文指定运行环境

统一实验环境如下：

| 配置项   | 大论文指定值              |
| -------- | ------------------------- |
| CPU      | Intel Xeon Platinum 8175M |
| GPU      | NVIDIA RTX A6000（48GB）  |
| 内存     | 128GB                     |
| 操作系统 | Ubuntu 20.04 LTS          |
| CUDA     | 12.1                      |
| PyTorch  | 2.1.0                     |

正式复现实验默认启用 `strict_environment: true`。程序启动时会检查操作系统、CPU、GPU、显存、系统内存、CUDA 和 PyTorch；任何一项不匹配都会在训练开始前终止运行。校验通过后，要求值、实际检测值及正式环境校验状态会写入 `provenance.json`。

同时Python 3.8.18、NumPy 1.24.4、SciPy 1.10.1、pandas 2.0.3  scikit-learn 1.3.2。

### 2.1 Conda 配置

必须在大论文指定的 Ubuntu 20.04、CPU、GPU 和内存硬件上执行：

```bash
conda env create -f environment.yml
conda activate biam-paper
```

`environment.yml` 固定 PyTorch 2.1.0 和 `pytorch-cuda=12.1`。

### 2.2 Docker 配置

`Dockerfile` 固定 Ubuntu 20.04、CUDA 12.1 和 PyTorch 2.1.0。Docker 不能模拟 CPU、GPU 型号或物理内存，因此宿主机仍必须是大论文中的 Intel Xeon Platinum 8175M、RTX A6000 48GB 和 128GB 内存。

```bash
docker build -t biam-paper .
docker run --rm --gpus all \
  -v "$PWD/results:/workspace/BIAM/results" \
  biam-paper
```

如果只需在其他环境检查代码兼容性，可显式加入 `--allow-environment-mismatch`。此开关只用于测试，`provenance.json` 中的 `formal_environment_validated` 会固定为 `false`，所得时间、显存和预测结果不得声称为大论文环境下的正式复现结果。

## 3. 快速验证

先运行缩小后的单种子回归流程，检查环境、训练、测试和结果落盘是否正常：

```bash
python biam_main.py --quick --task regression --output-dir results/quick
```

分类验证：

```bash
python biam_main.py --quick --task classification --output-dir results/quick
```

`--quick` 只用于冒烟测试，会缩小样本数、结构样本数、内层步数、训练轮数和随机种子数，其结果不能与论文表格比较。

在非论文环境中仅检查兼容性时：

```bash
python biam_main.py --quick --task regression \
  --allow-environment-mismatch \
  --output-dir results/quick
```

## 4. 论文配置实验

默认配置在 `configs/biam_default.yaml`。默认使用 5 个独立运行种子：

```yaml
seeds: [11, 22, 33, 44, 55]
```

运行默认仿真回归实验：

```bash
python biam_main.py --config configs/biam_default.yaml
```

运行仿真分类实验：

```bash
python biam_main.py \
  --config configs/biam_default.yaml \
  --task classification \
  --noise-ratio 0.2 \
  --imbalance-ratio 0.1 \
  --missing-mechanism MAR \
  --missing-ratio 0.3
```

也可显式指定 3 至 5 个种子：

```bash
python biam_main.py --seeds 101 202 303 404 505
```

同一设置中的所有对比方法应读取 BIAM 输出的 `split_indices.npz` 和 `data_artifacts.npz`，不要分别重新划分数据或生成扰动。论文表格采用 10 次重复时，可传入 10 个种子；本仓库根据当前复现要求默认运行 5 次并报告样本标准差。

## 5. 默认超参数

| 配置项                      |   默认值 | 含义                                                       |
| --------------------------- | -------: | ---------------------------------------------------------- |
| `hinge_bins`              |        8 | 每条曲线的 Hinge 结点数\(L_h\)，每个结点含正、反两个基函数 |
| `structure_samples`       |        8 | 反变量结构样本数\(S\)，必须是不小于 4 的偶数               |
| `inner_steps`             |        5 | 每个候选结构的下层更新步数\(K_{in}\)                       |
| `lambda_l0`               | `5e-4` | 预期结构组数量正则\(\lambda_0\)                            |
| `lambda_l2`               | `1e-4` | Hinge 与缺失效应系数正则\(\lambda_2\)                      |
| `lambda_kl`               | `1e-4` | Bernoulli 结构分布 KL 正则                                 |
| `prior_probability`       |      0.1 | 稀疏 Bernoulli 先验概率\(\pi_0\)                           |
| `gate_threshold`          |      0.5 | 部署硬门控阈值\(\kappa\)                                   |
| `min_interaction_support` |       20 | 有向交互的最小训练支持度\(n_{\min}\)                       |
| `noise_scale`             |      0.3 | 常规仿真回归基础噪声尺度\(\sigma_\varepsilon\)             |
| `lower_lr`                | `1e-2` | 可加模型更新学习率\(\eta_\theta\)                          |
| `structure_lr`            | `1e-2` | 结构分布更新学习率\(\eta_\phi\)                            |
| `weight_lr`               | `1e-3` | 权重网络更新学习率\(\eta_\psi\)                            |
| `batch_size`              |       64 | 训练批次大小                                               |
| `meta_batch_size`         |       64 | 元数据批次大小                                             |
| `epochs`                  |      200 | 最大训练轮数                                               |
| `patience`                |       20 | 调参集早停耐心值                                           |
| `final_refit_steps`       |      100 | 固定结构后的最终重拟合步数                                 |

常用命令行覆盖项：

```bash
python biam_main.py \
  --hinge-bins 12 \
  --structure-samples 8 \
  --inner-steps 5 \
  --epochs 200 \
  --patience 20 \
  --device cuda
```

## 6. 数据划分协议

### 6.1 仿真数据

每个种子独立生成完整数据，并按照 7:1:1:1 划分：

- `train`：下层加权拟合；
- `meta`：更新结构分布和权重网络；
- `tune`：选择超参数和早停位置；
- `test`：最终冻结模型后评价。

分类任务按类别分层；回归任务按无噪条件响应的秩分位组分层。默认仿真输入满足 \(X\sim N(0,\Sigma)\)、\(\Sigma_{jk}=0.5^{|j-k|}\)，响应和判别函数与 `biam.tex` 中的回归、分类仿真公式一致。

标签翻转、类别长尾下采样、基础回归噪声和响应异常只作用于训练集。仿真分类按 \(|g_i|\) 从小到大选择最靠近决策边界的训练样本翻转；外部分类数据采用对称随机翻转。元数据集、调参集和测试集使用干净目标。测试回归指标使用无异常条件响应。

### 6.2 真实或外部数据

外部数据使用 NPZ：

```python
import numpy as np

np.savez("dataset.npz", X=X, y=y)
```

分类任务：

```bash
python biam_main.py \
  --dataset npz \
  --data-path dataset.npz \
  --task classification \
  --num-classes 3
```

分类数据采用独立重复的 7:1:1:1 分层随机划分。回归数据采用 5 折外层交叉验证：每次留 1 折作为测试集，其余样本再按 7:1:1 划分为训练、元数据和调参集。程序先在一个种子的 5 个外层折上取指标平均，再跨种子计算均值和标准差，与论文附录三的汇总顺序一致。

Boston Housing、Plasma Retinol 等需要按训练集统计量标准化响应的任务使用：

```bash
python biam_main.py --dataset npz --data-path dataset.npz --task regression --standardize-target
```

响应均值和标准差只由当前外层折的训练集估计，并写入 `data_run.json`。

CME、ADNI 等带自然缺失的数据可直接在 `X` 中使用 `NaN`。模型最终缺失状态为自然缺失与人工缺失的逻辑并集。

### 6.3 缺失机制

人工缺失只作用于原本可观测的单元：

- 仿真数据候选特征为前一半特征；
- 外部数据由每个外层划分种子固定选择 `ceil(0.3 * p)` 个候选特征；
- MCAR 使用固定候选单元概率；
- MAR 使用不被人工遮蔽的锚点特征，斜率固定为 1.5；
- MNAR 使用当前特征的潜在标准化值，斜率绝对值固定为 1.5；
- MAR/MNAR 截距只在训练集上用二分法校准，随后冻结并应用于四个子集。

标准化统计量、中位数、Hinge 结点、效应中心化常数和候选交互也全部只由最终训练集估计。

## 7. 结果与复现文件

一次默认运行的目录结构如下：

```text
results/biam/regression_synthetic/
├── config.json
├── provenance.json
├── summary.json
└── seed_11/
    ├── seed_summary.json
    └── fold_0/
        ├── split_indices.npz
        ├── data_artifacts.npz
        ├── preprocessing.npz
        ├── data_run.json
        ├── training.json
        ├── metrics.json
        ├── predictions.csv
        └── model.pt
```

文件说明：

- `config.json`：实际完整超参数和种子列表；
- `provenance.json`：论文环境逐项校验、实际软硬件、Python/NumPy 版本、Git 提交及工作区状态；
- `split_indices.npz`：训练、元数据、调参和测试样本的原始索引；
- `data_artifacts.npz`：人工/自然缺失掩码、候选特征、MAR 锚点、MNAR 方向、校准截距、标签翻转或响应异常索引；
- `preprocessing.npz`：训练集中位数、均值、标准差、Hinge 结点、中心化常数、候选交互和支持度；
- `training.json`：早停轮数、逐轮训练/调参损失、缓存规模、最终结构概率和选择组；
- `predictions.csv`：测试样本原始索引、真实目标、逐样本预测及分类概率；
- `summary.json`：各随机种子的指标值、均值和样本标准差；
- `model.pt`：冻结后的模型、权重网络、结构参数和训练记录。

回归默认报告 MSE 与 \(R^2\)，分类默认报告 Accuracy 与 Macro-F1。测试集不参与早停、结构筛选、权重网络更新或最终参数重拟合。

## 8. 测试

```bash
python run_tests.py
```

测试覆盖四路索引互斥与复现、训练集专属扰动、缺失率校准、长尾和标签噪声、主效应/交互经验中心化、有向支持度筛选、硬组门控、反变量留一对基线，以及多种子完整结果落盘。

## 9. 复现注意事项

- 正式比较必须让所有方法共享完全相同的 `split_indices.npz`、标签/响应异常索引和缺失掩码。
- 不要在完整数据或测试集上重新估计标准化、中位数、Hinge 结点、候选交互或缺失概率截距。
- GPU 浮点规约、驱动和底层算子可能导致极小数值差异；代码统一设置 Python、NumPy、CPU/GPU PyTorch 随机状态，并启用确定性算法警告模式。
- 正式复现保持 `strict_environment: true`，不得使用 `--allow-environment-mismatch` 绕过环境检查。
- 惰性结构缓存按论文要求不淘汰。高维、长轮次运行时，缓存会随新结构访问增加；`training.json` 会记录每轮缓存规模。
