# CS5344 Project：面向异常检测的表格数据增强

本仓库是 NUS CS5344 Team 27（Offer-Squad）的共享工作区。项目目标不是提交 anomaly prediction，而是为两个网络流量数据集生成**仅代表正常（normal）流量**的合成训练数据，供 Kaggle 的固定异常检测器评测。

> 本文件也面向组员的 AI 助手：开始修改前，请阅读本 README、`Project_2627Sem1/` 下的课程材料和当前 Git 状态。不得尝试获取、搜索、重建或使用 Kaggle 的隐藏 validation/test 数据。

## Kaggle：已核实的正式要求

Kaggle MCP 已连接并于 2026-10-05 核实了两个 CS5344 competition。两场比赛都只允许使用已发布的带标签训练集开发方法；训练标签可用于方法设计，但提交文件不能包含标签或 prediction。

| Competition | 官方提交行数 | 评分指标 | 每日上限 |
|---|---:|---|---:|
| NSL-KDD | 40,000 | `(AUPRC_ECOD + mean(AUPRC_IForest over 10 fixed seeds)) / 2` | 5/team |
| UNSW-NB15 | 70,000 | `(AUPRC_ECOD + mean(AUPRC_IForest over 10 fixed seeds)) / 2` | 5/team |

提交 CSV 必须满足：

- 只包含意图为 normal 的样本；
- 第一列为连续且唯一的 `id`；
- 其余列的名称和顺序与官方模板完全一致；
- 不含 `is_anomaly` 或任何 prediction 列；
- categorical feature 由 Kaggle one-hot 编码后评测；
- hidden validation/test 不会向参赛者发布，public leaderboard 只能作为有限的开发反馈。

两场比赛截止时间均为 **2026-11-01 23:59:59（新加坡时间）**。参赛团队最多 3 人；所有代码与生成流程须在比赛结束后可复现。

## 当前提交记录

2026-10-05 已通过 Kaggle 提交 coverage v1，尚不应为公开榜分数反复调参：

| Competition | Kaggle submission ref | 文件 |
|---|---:|---|
| NSL-KDD | `56850612` | `submission-NSL-KDD-coverage-v1.csv` |
| UNSW-NB15 | `56850622` | `submission-UNSW-NB15-coverage-v1.csv` |

候选 CSV 保存在本地 `outputs/submissions/`，该目录被 Git 忽略，避免把每次实验生成的大文件纳入版本控制。

## 仓库内容

```text
.
├── Project_2627Sem1/                  # 课程提供的主要材料（请保持原样）
│   ├── CS5344_Project_Briefing_2627Sem1.pdf
│   ├── CS5344_anomaly_detection_problem_formulation.pdf
│   ├── CS5344_proposal_quiz_2627.pdf
│   └── starter_kit.py                 # 官方简单 baseline
├── train-NSL-KDD.csv                  # 30,000 行：20,000 normal / 10,000 anomaly
├── train-UNSW-NB15.csv                # 50,000 行：35,000 normal / 15,000 anomaly
├── eda_script.py                      # 早期 EDA 绘图脚本
├── src/
│   ├── baselines.py                   # full bootstrap 与 starter-style 对照
│   └── coverage_generator.py          # 主生成器与 schema 检查
├── scripts/
│   ├── evaluate_local.py              # 本地 ECOD + IForest proxy 评估
│   └── generate_submission.py         # 导出 Kaggle submission
├── requirements.txt
└── README.md
```

数据 schema：

- **NSL-KDD**：41 个 feature（38 数值、3 类别：`protocol_type`、`service`、`flag`）及 `is_anomaly`。
- **UNSW-NB15**：42 个 feature（39 数值、3 类别：`proto`、`service`、`state`）及 `is_anomaly`。
- 两份训练集均无缺失值；生成数据必须保持类别合法、二元列为 `0/1`、整数列为整数，并落在 normal 训练数据的观测数值范围内。

## 官方 baseline

`Project_2627Sem1/starter_kit.py` 是课程提供的格式/方法对照：从 20 个 normal prototype 重采样至 `TARGET_N=2000`，再对非二元数值列添加 Gaussian jitter。它容易遗漏罕见 normal mode 并破坏特征依赖关系，因此只应作为 baseline，而不是最终方法。

```bash
python3 Project_2627Sem1/starter_kit.py
```

## Coverage v1：当前主方法

主实现为 `src/coverage_generator.py`，入口为 `scripts/generate_submission.py`。它：

1. 使用全部 normal training rows；
2. 用三个类别 feature 的组合定义 observed normal signature；
3. 在经验频率与均匀覆盖之间分配输出配额，并限制罕见 signature 的最大过采样；
4. 在同一 signature 内混合 exact bootstrap 与有界数值插值；类别和二元 feature 不改变；
5. 导出前检查行数、`id`、列顺序、缺失/无穷值、类别、二元/整数约束及数值范围。

本地 proxy 使用 seed 42、80/20 分层 holdout 和正式提交预算。coverage v1 目前优于 starter-style jitter 和 full-normal bootstrap：

| Dataset | 生成预算 | coverage v1 local mean AUPRC |
|---|---:|---:|
| NSL-KDD | 40,000 | 0.9570 |
| UNSW-NB15 | 70,000 | 0.5017 |

这些是本地 proxy，不是 Kaggle 官方分数；官方 IForest 使用 10 个固定 seeds，hidden data 也不可访问。

导出命令：

```bash
python3 scripts/generate_submission.py \
  --train train-NSL-KDD.csv \
  --output outputs/submissions/submission-NSL-KDD-coverage-v1.csv \
  --target-n 40000 \
  --method coverage \
  --seed 42

python3 scripts/generate_submission.py \
  --train train-UNSW-NB15.csv \
  --output outputs/submissions/submission-UNSW-NB15-coverage-v1.csv \
  --target-n 70000 \
  --method coverage \
  --seed 42
```

本地对照评估：

```bash
python3 scripts/evaluate_local.py \
  --train train-NSL-KDD.csv \
  --target-n 40000 \
  --seed 42 \
  --output outputs/local_eval/nsl-kdd-target40000-seed42.json
```

## 协作约定

- 不直接覆盖 `train-*.csv`、课程 PDF 或他人的生成结果；新输出须包含数据集、方法、seed 和版本。
- 每个实验至少记录：数据集、切分、参数、generator seed、detector seed、AUPRC、输出路径和代码 commit。
- 不要围绕 public score 做无控制的参数搜索；先在固定本地切分选择少量候选，再审慎使用提交额度。
- 提交前运行 schema 检查；提交后记录 Kaggle submission ref、描述、时间和 public score。
- 新增依赖时同步更新 `requirements.txt`；不得提交缓存、虚拟环境、checkpoint 或 `outputs/` 下的中间文件。

## 参考材料

- `Project_2627Sem1/CS5344_anomaly_detection_problem_formulation.pdf`：正式任务定义、数据格式、feature 含义和隐藏 anomaly 类型设计。
- `Project_2627Sem1/CS5344_Project_Briefing_2627Sem1.pdf`：课程项目和 Kaggle 评测说明。
