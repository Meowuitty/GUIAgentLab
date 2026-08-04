# GUIAgentLab

[English](README.md) | **简体中文**

GUIAgentLab 提供 MAI-UI 在 MobileWorld 上的训练与评测代码。

## 方法

| 方法 | 训练信号 |
| --- | --- |
| **GRPO** | 根据完整 GUI 轨迹的任务结果计算组相对优势。 |
| **GiGPO** | 在轨迹结果之外，按照感知相似的 GUI 状态组织过程奖励。 |
| **ADMIRE-GRPO** | 根据任务里程碑、无效动作、循环和终局结果构造逐步奖励。 |
| **OPD** | 在训练过程中使用冻结教师模型进行在线策略蒸馏。 |
| **OPD + 教师干预** | 在无效动作、确定性循环和界面持续不变时，允许教师提供少量动作。 |

## 实验结果

在 105 个本地 MobileWorld 任务上进行 greedy 评测，每个任务生成一条轨迹。

| 模型 | 方法 | 成功率 |
| --- | --- | ---: |
| MAI-UI-2B | 原始模型 | 14.3% |
| MAI-UI-8B | 原始模型 | 31.4% |
| MAI-UI-2B | GRPO | 16.2% |
| MAI-UI-2B | GiGPO | 18.1% |
| MAI-UI-2B | GiGPO + 成功轨迹复用 | 20.0% |
| MAI-UI-2B | ADMIRE-GRPO | 17.1% |
| MAI-UI-2B | ADMIRE-GRPO + 成功轨迹复用 | 19.0% |
| MAI-UI-2B | OPD | 18.1% |
| MAI-UI-2B | OPD + 教师干预 | 19.0% |

## 快速开始

### 安装

```bash
git clone https://github.com/Meowuitty/GUIAgentLab.git
cd GUIAgentLab

conda create -n guiagentlab python=3.12 -y
conda activate guiagentlab

pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e ".[train]"
```

### 配置

```bash
cp .env.example .env
```

在 `.env` 中填写输出目录和模型路径：

```dotenv
GUIAGENTLAB_STATE_DIR=/path/to/guiagentlab-state
MODEL_PATH=/path/to/student-model
WANDB_API_KEY=your-key
```

OPD 还需要 `TEACHER_MODEL_PATH`，GiGPO 需要填写 `.env.example` 中的过程奖励
接口配置。

### 准备 MobileWorld

准备 MobileWorld 环境池：

```bash
docker pull ghcr.io/tongyi-mai/mobile_world:latest
./scripts/env/prepare_slots.sh 64
./scripts/env/recreate_pool.sh train
./scripts/validate/preflight.sh
```

### 训练

```bash
gal train grpo
gal train gigpo
gal train gigpo --replay success
gal train admire
gal train admire --replay success
gal train opd --teacher-model /path/to/teacher-model
gal train opd-teacher --teacher-model /path/to/teacher-model
```

可以直接覆盖模型和数据路径：

```bash
gal train grpo \
  --model /path/to/student-model \
  --train-file /path/to/train.parquet \
  --validation-file /path/to/test.parquet
```

### 评测

运行 105 题评测：

```bash
ACTIVE_CONTAINERS=56 SPARE_CONTAINERS=8 \
  ./scripts/env/recreate_pool.sh rollout

MODEL_PATH=/path/to/checkpoint \
  ./scripts/eval/maiui.sh my-model
```

结果写入
`$GUIAGENTLAB_STATE_DIR/evaluations/<run-id>/results/summary.json`。

## 内置数据

| 路径 | 内容 |
| --- | --- |
| `data/train.parquet` | 60 行加权训练数据，覆盖 35 个任务，供 GRPO、GiGPO 和 ADMIRE 使用。 |
| `data/opd.parquet` | OPD 使用的 30 行训练数据。 |
| `data/test.parquet` | 105 个本地 MobileWorld 评测任务，只用于评测。 |
| `data/replay/success/` | 18 条成功轨迹及其里程碑和过程奖励缓存。 |

## 项目结构

```text
GUIAgentLab/
├── guiagentlab/       # 方法、轨迹复用、rollout、奖励和集成代码
├── configs/           # 训练配置和固定的上游版本
├── data/              # 固定数据集和成功轨迹
├── scripts/           # 环境、训练、验证和评测入口
├── mobile_world/
└── verl/
```
