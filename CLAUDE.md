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

# Type check (env pointed via pyproject.toml [tool.basedpyright] venvPath/venv — NOT pythonPath, that key is silently rejected)
basedpyright

# Run a single test file
pytest Tests/path/to/test_file.py

# View training metrics
tensorboard --logdir=logs
```

### Testing

Most tests build a real `StackGanGenModel`/`StackGanDiscModel` (GPU inference against StackGan2 checkpoints). Root `conftest.py` has session-scoped fixtures (`facade_gen_model`, `facade_gen_model_scale0`, `facade_disc_model`) — reuse these instead of constructing models per-test (expensive). `require_path()` in `conftest.py` skips a test cleanly when a gitignored external asset (StackGan2 checkpoints/configs, `Tests/*/images/`) isn't present, instead of hard-crashing. Pytest config (`testpaths`, `pythonpath`) lives in `pyproject.toml`.

## Architecture Overview

This is a **preference-based reinforcement learning system** for image generation. The agent learns to navigate a generative model's noise space using pairwise human (or artificial) preferences to find noise vectors that produce images matching a desired target. The system is implemented around StackGAN v2 and is currently configured for the CelebA dataset.

### The Training Loop (`src/Pipeline/PbRLPipeline`)

`main.py` only wires components and calls `pipeline.run()`; the loop itself lives in `PbRLPipeline._run_round()`, called once per round from `run()`, followed by `_run_control_logging()` after all rounds. Each round:
1. **Sample** noise vectors (`ActionData`) from the action distribution
2. **Filter** candidates using the current reward model
3. **Generate** image pairs from candidate actions via the generative model
4. **Collect preferences** (human via Tkinter GUI, or artificial via cosine similarity)
5. **Store** preference pairs in memory with exponential discount weighting
6. **Train reward model** to predict pairwise preferences (cross-entropy loss)
7. **Optimize** the destination action (the "best" noise vector) by maximizing predicted reward
8. Repeat

### Source Layout

**`src/Abstract/`** — Interfaces most components implement, each with a `Configuration` dataclass + `create_from_configuration` (e.g. `AbsRewardModel`, `AbsFeedbackSource`, `AbsActionFilter`). Exceptions: `AbsGenModel`, `AbsDiscModel`, `AbsMemory` exist but their concrete classes (`StackGanGenModel`, `StackGanDiscModel`, `RoundsMemory`) do **not** inherit them — treat these three as documentation of intended shape, not an enforced contract.

**`src/Pipeline/`** — `PbRLPipeline` encapsulates the round loop and post-loop control logging (see Training Loop above). Caller (`main.py`) is responsible for constructing components with shared live-object dependencies (e.g. `destination_action_trainable`) and passing them in already wired; standalone helpers with no such cross-object dependencies are constructed internally.

**`src/DataStructures/`** — Typed tensor wrappers with shape validation:
- `ActionData [B, D]` — noise vectors fed to the generator
- `ActionPairsData [B, 2, D]` — pairs of actions for comparison
- `PreferencePairsData` — labels: `[1,0]` left, `[0,1]` right, `[0.5,0.5]` equal, `[0,0]` skip
- `ImageData [N, C, H, W]` — generated images
- `TrainableActionData` — wraps a live `nn.Parameter` for `ptLightningLatentWrapper`'s latent optimization. `.actions` aliases `.grad_actions` (grad-tracked); `.detached_actions` is the safe non-training read. `clone()`/`append()` intentionally raise — it's a single shared autograd-graph object, not meant to be copied.
- `ActionPairsPrefPairsContainer` — bundles an `ActionPairsData`/`PreferencePairsData` pair for `RoundsMemory.add_data()`.

**`src/GenModel/`** — Wraps StackGAN v2. `StackGanGenModel` provides `generate()`, `sample_random_actions()`, `get_input_noise_distribution()`, and `SetDevice()` for CUDA/CPU switching.

**`src/DiscModel/`** — Wraps the StackGAN v2 discriminator. `StackGanDiscModel` mirrors `StackGanGenModel`'s structure (`discriminate()`, `SetDevice()`, same `DataParallel`/`ngpu` device wiring).

**`src/RewardModel/`** — `mlpRewardNetwork`: 3-layer MLP with LeakyReLU/dropout that maps a noise vector to a scalar reward.

**`src/Trainer/`** — `ptLightningTrainer` is a generic Lightning runner; which model it trains depends on the wrapper (`ptLightningWrappers.py`) it's given:
- `ptLightningModelWrapper` — trains the reward model
- `ptLightningLatentWrapper` — treats the destination action as a learnable parameter and optimizes it (reward model is frozen during this step)

**`src/FeedbackSource/`** — `HumanFeedback` (Tkinter GUI), `CosDistFeedback` (cosine similarity to a target image), `RandomFeedbackSource`.

**`src/PreferenceDataGenerator/`** — `GraphPreferenceDataGeneration` uses NetworkX to build a directed preference graph and infer transitive preferences. `BestActionTracker` decorates this to track the overall best action across rounds.

**`src/Memory/`** — `RoundsMemory` stores the last N rounds of `(action_pairs, preferences)` and applies `discount_factor^age` weighting when returning all data for training.

**`src/ActionDistribution/`** — `SimpleActionDistribution` (pure random from generator prior), `GreedyNormalActionDistribution` (epsilon-greedy Gaussian around the current destination action).

**`src/Filter/`** — `ScoreActionFilter` selects top/bottom-N actions by predicted reward; `CompositeActionFilter` (file `CompositeSeriesActionFilter.py` — file/class names differ) chains multiple filters; `EmptyActionFilter` (pass-through) and `RandomActionFilter` (random N) round out the basic filters. `UncertaintyActionFilter` is an intentional non-functional placeholder for a future feature (constructor always raises `NotImplementedError`, references removed `AbsNetworkExtension` API) — leave it, don't remove as dead code.

**`src/Loss/`** — `PreferenceLoss` (cross-entropy on softmax-ed reward pairs), `ActionRewardLoss` (negated reward for maximization), `LogLossDecorator`, `CompositeLoss`, `RewardValuesRegularisation`, `ActionDiscriminatorLoss`.

**`src/MetricsLogger/`** — `TensorboardImageLogger`, `TensorboardGridImageLogger`, `TensorboardScalarLogger`, `CompositeLogger`.

### Key Design Conventions

A class inheriting both an `src/Abstract` base and `nn.Module` (e.g. `TrainableActionData(ActionData, nn.Module)`) must call `nn.Module.__init__(self)` explicitly in `__init__` — `AbsData`/`ActionData` don't chain `super().__init__()`, so skipping it raises `AttributeError: cannot assign parameters before Module.__init__() call` the moment you assign an `nn.Parameter`.

`ActionRewardLoss`/`mlpRewardNetwork` read `data.actions` generically across any `ActionData` subtype. A subclass needing different `.actions` semantics (e.g. grad passthrough) must alias the property to what it wants those generic readers to see — blocking or renaming it away breaks the loss-calculation chain with no type error, only a failing training run.

Inside `src/DataStructures/`, import sibling classes directly from their submodule (`from .X import X`), never through the package itself (`from src.DataStructures import X`) — `__init__.py`'s import order can then silently bind the wrong object (a bare submodule instead of the class) with no static-analysis or immediate-runtime signal.

`torch.tensor` (function) vs `torch.Tensor` (class) — only use `torch.Tensor` in type annotations, `torch.tensor(...)` is for constructing tensors. Easy typo, basedpyright catches it as "Expected class but received function".

Use `lightning.pytorch.*` imports everywhere, never `pytorch_lightning.*` — both packages are installed but are different classes at runtime (`Logger`, `Callback`, etc. don't match), breaking Lightning's internal isinstance checks despite looking interchangeable.

Abstract base classes in `src/Abstract/` must declare the exact param names/types every concrete override uses (basedpyright flags LSP-violating narrowing otherwise). `AbsLoss` is `Generic[D]` since different loss subclasses need different input data types — subclass as `AbsLoss[ActionData]`, not bare `AbsLoss`.

`StackGanGenModel`/`StackGanDiscModel` wrap their model in `torch.nn.DataParallel(module, list(range(ngpu)))` at construction — if `ngpu==1`, `DataParallel.__init__` auto-moves the module to `cuda:0` right there (before `load_state_dict`), regardless of the `self.device="cpu"` bookkeeping value set alongside it. `DataParallel.forward()` requires every param to sit on `device_ids[0]` whenever `device_ids` is non-empty, but skips that check entirely (calls the module directly) when `device_ids` is empty — so `SetDevice()` must keep `device_ids`/`src_device_obj` in sync with the target device (cleared for CPU, `range(primary_idx, primary_idx+ngpu)` for CUDA), not just call `.to(device)`.

Every component follows the same pattern:
1. Inherits from an abstract base class in `src/Abstract/`
2. Has a nested `Configuration` dataclass for its parameters
3. Exposes a `create_from_configuration(cfg)` static method

This means adding a new component requires: writing the class + abstract base (if new category) + `Configuration` dataclass + wiring it into `main.py` (components are wired by hand there — no config-driven builder).

### External Dependencies

StackGAN v2 model files are **not in the repository**. They must be placed in `GenerativeModelsData/StackGan2/` before running the pipeline:
- YAML config (e.g., `facade_3stages_color.yml`)
- Pre-trained generator checkpoint (`netG_*.pth`)
- Pre-trained discriminator checkpoint (`netD*.pth`)

Paths to these files are currently hardcoded in `main.py`.
