> This project is not intact since some scripts may be missing or incomplete. Review the manuscript `manuscript.tex` for the complete details of the research. Review all the project and supply relative necessary scripts to ensure the project can be executed successfully.

## Summary of Changes

Here is everything that was done to make the project executable:

### 1. Added `MelSpectrogramDataset` class (image_classification.py)
The advanced scripts (ablation_study.py, sota_comparison.py, hyperparam_sensitivity.py, noise_robustness.py) referenced `MelSpectrogramDataset` which didn't exist. Added a flat image dataset class that scans class directories and supports cross-validation splits, noise injection, and custom transforms.

### 2. Fixed broken imports across 4 scripts

| Script | Broken Import | Fixed To |
|--------|--------------|----------|
| ablation_study.py | `from ...pinn_alexnet import PINNAlexNet` | Removed (unused) |
| ablation_study.py | `from ...train_eval import train_epoch` | `train_one_epoch` |
| sota_comparison.py | `from ...pinn_alexnet import PINNAlexNet` | Removed (unused) |
| sota_comparison.py | `from ...pinn_convnext import PINNConvNeXt` | `PINNConvNeXtClassifier` |
| hyperparam_sensitivity.py | `from ...pinn_alexnet import PINNAlexNet` | Removed (unused) |
| noise_robustness.py | `from ...pinn_alexnet import PINNAlexNet` | Removed (unused) |

### 3. Implemented physical field feedback (manuscript Section 4.4)
The manuscript describes the physical field $\Phi$ being fed back to the classifier. This was missing from several model implementations. Fixed in:
- pinn_alexnet.py — `PINNAlexNetClassifier`
- ablation_study.py — `AlexNetWithPINN`
- sota_comparison.py — `ProposedModel`
- hyperparam_sensitivity.py — `SensitivityModel`
- noise_robustness.py — `PINNModel`

### 4. Added `--split_only` mode to build_dataset.py
Enables splitting an existing ImageFolder-style directory (e.g., after PCNN processing) without rebuilding from raw audio.

### 5. Created convenience pipeline script (run_pipeline.sh)
One-command pipeline: raw WAV → Mel-GADF images → PCNN → train/val/test split → training. Supports `--quick`, `--skip-build`, `--skip-train` flags.

### 6. Created ablation.yaml
Missing config file referenced by ablation_study.py.

### 7. Installed missing `tensorboard` dependency
Was listed in requirements.txt but not installed on the system.

### 8. Updated README.md
Corrected dataset construction workflow, added advanced experiments section, and updated model specifications.

### Verified pipeline
- ✅ All 28 scripts compile cleanly
- ✅ All core library imports work
- ✅ All 5 model architectures forward-pass correctly
- ✅ Config loading works
- ✅ Dataset building from raw WAV files works (tested: 651 fused images generated in ~71s)

Made changes.


>check the `scripts` directory and remove unnecessary and redundant scripts in this project.

**Removed 18 redundant legacy scripts:**

| Category | Scripts Removed | Superseded By |
|----------|----------------|---------------|
| Mel-GAF variants | `Mel-GAF-0..4.py`, Mel-GAF.py (6) | build_dataset.py + data-optimization |
| Old PCNN processors | PCNN.py, PCNN_1.py (2) | apply_pcnn.py + pcnn.py |
| Monolithic PCNN-PINN trainers | PCNN-PINN-AlexNet.py, `-LeNet.py`, `-ResNet.py`, `-ResNet-phys.py`, `-VGGNet.py` (5) | Modular src + train.py |
| Typo variant | `PCNN-PIINN-ResNet-phys(original).py` (1) | — |
| Monolithic PINN trainers | `PINN-AlexNet.py`, `-LeNet.py`, `-ResNet.py`, `-VGGNet.py` (4) | Modular src + train.py |

**10 core scripts remain:**

| Script | Purpose |
|--------|---------|
| build_dataset.py | Raw WAV → Mel-GADF fused images + split |
| apply_pcnn.py | PCNN enhancement of fused images |
| train.py | Train any PINN backbone via config |
| evaluate.py | Evaluate trained model (metrics + plots) |
| visualize.py | Training curves + comparison charts |
| run_experiments.py | Batch experiment runner |
| ablation_study.py | Component-wise ablation (Table 6) |
| sota_comparison.py | SOTA benchmarking (Table 7) |
| hyperparam_sensitivity.py | Hyperparameter sensitivity analysis |
| noise_robustness.py | Noise robustness evaluation |


>update the `README.md` according to the `manuscript.tex` and all scripts in the project, also add some experimental results and figures to the `README.md` to make it more informative and visually appealing.

### What changed

| Aspect | Before | After |
|--------|--------|-------|
| **Title** | Generic "PINN for Transformer Fault Diagnosis" | Matches manuscript: "PCNN-Enhanced Multi-Representation Fusion..." |
| **Results** | Non-manuscript numbers (98.48%, 45.2M params) | **All 4 tables aligned with manuscript**: Ablation (Table 6), Backbone (Table 5), GASF vs GADF (Table 4), SOTA (Table 7) |
| **Figures** | None | **11 embedded images**: 5× multi-condition fused representations, 4× PCNN threshold comparison, 2× gamma correction comparison |
| **Structure** | ~200 lines, outdated | **435 lines**, complete project tree with all 10 scripts + full src library |
| **Scripts** | Only 6 documented, old names | All 10 current scripts documented with purpose & usage |
| **Pipeline** | Incomplete/confusing steps | Clear dual-path (automated run_pipeline.sh OR step-by-step) |
| **Models** | Wrong param counts, 1-channel | Correct params, 3-channel RGB input, physical feedback noted |
| **Extras** | Missing | Mermaid diagram, innovation table, evaluation outputs, funding info, model architecture detail |