# GUIAgentLab

**English** | [简体中文](README.zh-CN.md)

GUIAgentLab contains the training and evaluation code for MAI-UI agents on
MobileWorld.

## Methods

| Method | Training signal |
| --- | --- |
| **GRPO** | Group-relative advantages from complete GUI trajectory outcomes. |
| **GiGPO** | Trajectory outcomes plus process rewards grouped by perceptually similar GUI states. |
| **ADMIRE-GRPO** | Step rewards derived from task milestones, invalid actions, loops, and terminal outcomes. |
| **OPD** | Online policy distillation from a frozen teacher during optimization. |
| **OPD + teacher intervention** | OPD with a small teacher-action budget for invalid actions, deterministic loops, and repeated no-change states. |

## Results

Greedy evaluation on 105 local MobileWorld tasks, with one trajectory per task.

| Model | Method | Success rate |
| --- | --- | ---: |
| MAI-UI-2B | Original | 14.3% |
| MAI-UI-8B | Original | 31.4% |
| MAI-UI-2B | GRPO | 16.2% |
| MAI-UI-2B | GiGPO | 18.1% |
| MAI-UI-2B | GiGPO + Success Replay | 20.0% |
| MAI-UI-2B | ADMIRE-GRPO | 17.1% |
| MAI-UI-2B | ADMIRE-GRPO + Success Replay | 19.0% |
| MAI-UI-2B | OPD | 18.1% |
| MAI-UI-2B | OPD + teacher intervention | 19.0% |

## Quick start

### Install

```bash
git clone https://github.com/Meowuitty/GUIAgentLab.git
cd GUIAgentLab

conda create -n guiagentlab python=3.12 -y
conda activate guiagentlab

pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e ".[train]"
```

### Configure

```bash
cp .env.example .env
```

Set the output directory and model path in `.env`:

```dotenv
GUIAGENTLAB_STATE_DIR=/path/to/guiagentlab-state
MODEL_PATH=/path/to/student-model
WANDB_API_KEY=your-key
```

OPD also needs `TEACHER_MODEL_PATH`. GiGPO needs the process-reward API fields
shown in `.env.example`.

### Prepare MobileWorld

Prepare the MobileWorld pool:

```bash
docker pull ghcr.io/tongyi-mai/mobile_world:latest
./scripts/env/prepare_slots.sh 64
./scripts/env/recreate_pool.sh train
./scripts/validate/preflight.sh
```

### Train

```bash
gal train grpo
gal train gigpo
gal train gigpo --replay success
gal train admire
gal train admire --replay success
gal train opd --teacher-model /path/to/teacher-model
gal train opd-teacher --teacher-model /path/to/teacher-model
```

Paths can be overridden from the command line:

```bash
gal train grpo \
  --model /path/to/student-model \
  --train-file /path/to/train.parquet \
  --validation-file /path/to/test.parquet
```

### Evaluate

Run the 105-task evaluation:

```bash
ACTIVE_CONTAINERS=56 SPARE_CONTAINERS=8 \
  ./scripts/env/recreate_pool.sh rollout

MODEL_PATH=/path/to/checkpoint \
  ./scripts/eval/maiui.sh my-model
```

Results are written to
`$GUIAGENTLAB_STATE_DIR/evaluations/<run-id>/results/summary.json`.

## Included data

| Path | Contents |
| --- | --- |
| `data/train.parquet` | 60 weighted rows over 35 tasks, shared by GRPO, GiGPO, and ADMIRE. |
| `data/opd.parquet` | The 30-row dataset used by OPD. |
| `data/test.parquet` | All 105 local MobileWorld tasks, used only for evaluation. |
| `data/replay/success/` | 18 successful trajectories with milestone and process-reward caches. |

## Repository layout

```text
GUIAgentLab/
├── guiagentlab/       # first-party methods, replay, rollout, rewards, and integration
├── configs/           # training configuration and pinned upstream revisions
├── data/              # fixed datasets and the 18-trajectory replay package
├── scripts/           # environment, training, validation, and evaluation entrypoints
├── mobile_world/
└── verl/
```
