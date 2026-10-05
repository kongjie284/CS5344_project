# CS5344 Project：面向异常检测的表格数据增强

本仓库是 NUS CS5344 小组项目的共享工作区。目标不是提交分类器预测结果，而是为两个网络流量数据集生成 **仅表示正常（normal）流量** 的合成训练数据，供 Kaggle 的固定异常检测器评测。

> 本文件也面向 AI 助手：开始修改代码前，请先阅读本 README、`Project_2627Sem1/` 下的课程说明，以及当前工作区的 Git 状态。不要修改训练数据、课程原始 PDF 或已生成 submission，除非任务明确要求。

## 项目任务与 Kaggle 评分

每个数据集都提供了带 `is_anomaly` 标签的训练集：`0` 为 normal、`1` 为 anomaly。我们可以在设计生成策略时使用所有训练特征和标签，但最终上传的 CSV：

- 必须只包含意图为 normal 的样本；
- 必须有 `id` 列，其余列严格遵循训练集的特征列顺序；
- **不能**包含 `is_anomaly`；
- 行数必须严格等于 Kaggle competition 页面或 `sample_submission.csv` 指定的数量。

Kaggle 会用提交的合成 normal 数据训练固定 detector，并在隐藏验证/测试数据上评测：

`score = (AUPRC_ECOD + AUPRC_IsolationForest) / 2`

其中 ECOD 是基于经验尾部/排序的异常检测器；Isolation Forest 使用课程固定的配置。隐藏集包含训练中未出现的异常类型，因此公开榜只能作为开发反馈，不能为公开榜过拟合。

### 关于提交行数

仓库现有的 starter kit 和两份 submission 都生成 **2,000 行**，适合作为开发阶段 baseline。团队 proposal 曾规划最终导出 70,000 行。两者都不能替代 Kaggle 的实时官方要求：最终运行前，必须以 competition 页面或官方 sample submission 的行数为唯一依据，并在生成脚本中显式设置 `TARGET_N`。

## 仓库内容

```text
.
├── Project_2627Sem1/                 # 课程提供的主要材料（请保持原样）
│   ├── CS5344_Project_Briefing_2627Sem1.pdf
│   ├── CS5344_anomaly_detection_problem_formulation.pdf
│   ├── CS5344_proposal_quiz_2627.pdf
│   └── starter_kit.py                # 官方简单 baseline
├── train-NSL-KDD.csv                 # 30,000 行：20,000 normal / 10,000 anomaly
├── train-UNSW-NB15.csv               # 50,000 行：35,000 normal / 15,000 anomaly
├── submission-NSL-KDD.csv            # 当前 2,000 行 baseline submission
├── submission-UNSW-NB15.csv          # 当前 2,000 行 baseline submission
├── eda_script.py                     # EDA：模式坍缩 t-SNE 与偏态特征分布图
├── eda_nsl-kdd.png                   # 已生成的 NSL-KDD EDA 图
├── eda_unsw-nb15.png                 # 已生成的 UNSW-NB15 EDA 图
├── CS5344_Project_Proposal-Team27.pdf # 已提交 proposal，仅作背景参考
└── README.md
```

数据 schema：

- **NSL-KDD**：41 个特征（38 数值、3 类别：`protocol_type`、`service`、`flag`）与 `is_anomaly`。
- **UNSW-NB15**：42 个特征（39 数值、3 类别：`proto`、`service`、`state`）与 `is_anomaly`。
- 两份训练集均没有缺失值；类别列必须保留合法字符串取值，二元列必须保持为 `0/1`。

## 当前 baseline

`Project_2627Sem1/starter_kit.py` 的流程：

1. 仅保留 normal 行；
2. 固定抽取 20 个 prototype；
3. 有放回重采样到 `TARGET_N=2000`；
4. 对非二元数值列加入独立 Gaussian jitter，并裁剪回观测范围；
5. 输出 `id + feature columns`。

从仓库根目录运行：

```bash
python Project_2627Sem1/starter_kit.py
```

修改脚本顶部的 `TRAIN_CSV` 和 `OUTPUT_CSV` 可以在两个数据集间切换。它是**格式正确的最低基线**，但 20 个 prototype 容易遗漏罕见 normal 模式，独立加噪也可能破坏数值特征之间的依赖关系；不建议把它当作最终方法。

## 团队主生成器（开发中）

主方法位于 `src/coverage_generator.py`，命令入口为 `scripts/generate_submission.py`。第一版采用：

- 使用全部 normal 样本，而不是只取 20 个 prototype；
- 以三个类别特征组合（categorical signature）作为正常运行模式；
- 在经验频率与均匀覆盖之间分配生成配额，同时限制罕见模式的最大过采样倍数；
- 在相同类别签名内混合 exact bootstrap 与小幅数值插值；类别和二元特征保持不变；
- 自动验证行数、ID、列顺序、缺失值和无穷值。

示例（`2000` 仅用于本地开发；最终值必须换成官方要求）：

```bash
python3 scripts/generate_submission.py \
  --train train-NSL-KDD.csv \
  --output /tmp/submission-nsl-kdd-coverage.csv \
  --target-n 2000 \
  --seed 42
```

这只是首版可复现方法，尚未经过本地 detector 对照实验挑选参数，也尚未等同于最终 Kaggle 方法。

## 推荐的完成路线

建议保留 starter kit 作为对照，并新增独立、可复现实验代码，而不是直接把复杂逻辑堆进 starter kit。

1. **确定官方约束**：从 Kaggle 页面记录最终 `TARGET_N`、列顺序、submission 次数限制与任何最新规则。
2. **建立本地评估**：对已发布数据做固定的、按标签分层的开发/审计切分；每次方法比较均以相同切分、随机种子和预算运行 ECOD 与 Isolation Forest，分别报告 AUPRC 与平均值。
3. **先完成强对照**：全体 normal bootstrap、starter-kit 20 prototype + jitter、全体 normal + jitter。
4. **实现主要 generator**：按 normal 数据的类别签名或混合表示的 cluster 分配生成配额；从兼容的局部邻居中受限插值；保留一部分 bootstrap 以维持观察到的离散模式和尾部。任何 anomaly-aware rejection/screening 应作为独立消融组件。
5. **做数据质量检查**：列顺序、无缺失/无无穷、数值范围、整数/二元约束、类别合法性、重复率、类别签名覆盖、分位数/零值比例、相关性偏差。
6. **参数选择与复现**：不要用 public leaderboard 直接调参；保留配置、随机种子、运行命令、指标和生成审计日志。最终冻结配置后重新用全部训练数据生成 Kaggle submission。

建议后续代码结构：

```text
src/
  data.py          # 读取、schema、特征类型与约束
  baselines.py     # starter / full-bootstrap / jitter 对照
  generator.py     # 主生成器
  validation.py    # 输出质量与 submission schema 检查
  evaluate.py      # 本地 ECOD + Isolation Forest 评估
scripts/
  run_experiment.py
  generate_submission.py
configs/
  nsl_kdd.yaml
  unsw_nb15.yaml
outputs/           # 忽略大型中间产物；保留配置和关键指标摘要
```

## 协作约定

- 不直接覆盖 `train-*.csv`、课程 PDF 或他人生成的最终 submission；新输出请使用带数据集、方法、种子和时间/版本的文件名。
- 每个可比较实验至少记录：数据集、切分、方法/参数、generator seed、detector seed、AUPRC、输出路径和代码 commit。
- 提交前使用独立验证函数检查 submission 行数、`id` 范围、列名/顺序、缺失值、无穷值和数据类型约束。
- 提交代码前先执行 `git status`；避免把缓存、虚拟环境、模型 checkpoint 或大型临时结果提交到仓库。
- 如果新增依赖，请同步更新 `requirements.txt` 或等效环境说明，并在 README 补充运行方式。

## 参考材料

- `Project_2627Sem1/CS5344_anomaly_detection_problem_formulation.pdf`：正式任务定义、数据格式、特征含义和隐藏异常类型设计。
- `Project_2627Sem1/CS5344_Project_Briefing_2627Sem1.pdf`：Kaggle 评分流程与课程评估说明。
- `CS5344_Project_Proposal-Team27.pdf`：团队已提交的研究动机和初步方法思路；不是当前实现的唯一约束。
