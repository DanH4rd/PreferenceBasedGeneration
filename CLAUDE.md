# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Create/activate environment (conda, not pip — see environment.yml)
conda env create -f environment.yml
conda activate pbrl2

# Run the main pipeline
python main.py

# Run tests
pytest Tests/

# Lint and format
ruff check .
ruff format .

# Run a single test file
pytest Tests/path/to/test_file.py

# View training metrics
tensorboard --logdir=logs
```

### Testing

Most tests build a real `StackGanGenModel`/`StackGanDiscModel` (GPU inference against StackGan2 checkpoints). Root `conftest.py` has session-scoped fixtures (`facade_gen_model`, `facade_gen_model_scale0`, `facade_disc_model`) — reuse these instead of constructing models per-test (expensive). `require_path()` in `conftest.py` skips a test cleanly when a gitignored external asset (StackGan2 checkpoints/configs, `Tests/*/images/`) isn't present, instead of hard-crashing. Pytest config (`testpaths`, `pythonpath`) lives in `pyproject.toml`.

## Architecture Overview

This is a **preference-based reinforcement learning system** for image generation. The agent learns to navigate a generative model's noise space using pairwise human (or artificial) preferences to find noise vectors that produce images matching a desired target. The system is implemented around StackGAN v2 and is currently configured for the CelebA dataset.

### The Training Loop (`main.py`)

Each round of the pipeline:
1. **Sample** noise vectors (`ActionData`) from the action distribution
2. **Filter** candidates using the current reward model
3. **Generate** image pairs from candidate actions via the generative model
4. **Collect preferences** (human via Tkinter GUI, or artificial via cosine similarity)
5. **Store** preference pairs in memory with exponential discount weighting
6. **Train reward model** to predict pairwise preferences (cross-entropy loss)
7. **Optimize** the destination action (the "best" noise vector) by maximizing predicted reward
8. Repeat

### Source Layout

**`src/Abstract/`** — Interfaces that all major components implement. Every component has a corresponding abstract base class here (e.g. `AbsGenModel`, `AbsRewardModel`, `AbsFeedbackSource`, `AbsMemory`, `AbsActionFilter`).

**`src/DataStructures/`** — Typed tensor wrappers with shape validation:
- `ActionData [B, D]` — noise vectors fed to the generator
- `ActionPairsData [B, 2, D]` — pairs of actions for comparison
- `PreferencePairsData` — labels: `[1,0]` left, `[0,1]` right, `[0.5,0.5]` equal, `[0,0]` skip
- `ImageData [N, C, H, W]` — generated images

**`src/GenModel/`** — Wraps StackGAN v2. `StackGanGenModel` provides `generate()`, `sample_random_actions()`, and `get_input_noise_distribution()`.

**`src/RewardModel/`** — `mlpRewardNetwork`: 3-layer MLP with LeakyReLU/dropout that maps a noise vector to a scalar reward.

**`src/Trainer/`** — PyTorch Lightning wrappers:
- `ptLightningTrainer` — trains the reward model
- `ptLightningLatentWrapper` — treats the destination action as a learnable parameter and optimizes it (reward model is frozen during this step)

**`src/FeedbackSource/`** — `HumanFeedback` (Tkinter GUI), `CosDistFeedback` (cosine similarity to a target image), `RandomFeedbackSource`.

**`src/PreferenceDataGenerator/`** — `GraphPreferenceDataGeneration` uses NetworkX to build a directed preference graph and infer transitive preferences. `BestActionTracker` decorates this to track the overall best action across rounds.

**`src/Memory/`** — `RoundsMemory` stores the last N rounds of `(action_pairs, preferences)` and applies `discount_factor^age` weighting when returning all data for training.

**`src/ActionDistribution/`** — `SimpleActionDistribution` (pure random from generator prior), `GreedyNormalActionDistribution` (epsilon-greedy Gaussian around the current destination action).

**`src/Filter/`** — `ScoreActionFilter` selects top/bottom-N actions by predicted reward; `CompositeSeriesActionFilter` chains multiple filters.

**`src/Loss/`** — `PreferenceLoss` (cross-entropy on softmax-ed reward pairs), `ActionRewardLoss` (negated reward for maximization), `LogLossDecorator`, `CompositeLoss`.

**`src/MetricsLogger/`** — `TensorboardImageLogger`, `TensorboardScalarLogger`, `CompositeLogger`.

**`builder/`** — `StandardBuilder` is a factory that constructs components from `Configuration` dataclasses; `CfgBuilder` loads those configs from YAML (`builder/cfg/basic.yml`).

### Key Design Conventions

Every component follows the same pattern:
1. Inherits from an abstract base class in `src/Abstract/`
2. Has a nested `Configuration` dataclass for its parameters
3. Exposes a `create_from_configuration(cfg)` static method

This means adding a new component requires: writing the class + abstract base (if new category) + `Configuration` dataclass + wiring it into `StandardBuilder` and `main.py`.

### External Dependencies

StackGAN v2 model files are **not in the repository**. They must be placed in `GenerativeModelsData/StackGan2/` before running the pipeline:
- YAML config (e.g., `facade_3stages_color.yml`)
- Pre-trained generator checkpoint (`netG_*.pth`)
- Pre-trained discriminator checkpoint (`netD*.pth`)

Paths to these files are currently hardcoded in `main.py`.
