# DeformationInterpretation — InSAR Deformation Interpretation

A supervised deep-learning framework for classifying and localising coseismic surface deformation from InSAR interferograms. Three backbone families are implemented and compared: **CNN** (ResNet), **ViT** (Vision Transformer), and **ViM** (Vision Mamba).
> General implementation of **"Automated deformation detection and interpretation using InSAR data and a multi-task ViT model"**  
> Abdallah, M., Younis, S., Wu, S., & Ding, X. (2026). *International Journal of Applied Earth Observation and Geoinformation*, 128, 103758  .  
> [10.1016/j.jag.2024.103758](https://doi.org/10.1016/j.jag.2024.103758)

---

## Overview

Interferometric Synthetic Aperture Radar (InSAR) produces phase maps (interferograms) that capture surface displacements caused by earthquakes. This project trains deep networks to:

1. **Classify** the fault mechanism from a wrapped or unwrapped interferogram into one of up to four categories.
2. **Localise** the deformation region by regressing a bounding-box.

Both tasks can be learned jointly or independently through shared backbone representations.

---

## Repository structure

```
├── CNN/
│   ├── train.ipynb          # Training entry-point for the CNN model
│   └── Network/
│       └── src/
│           ├── model.py     # CNN model (ResNet backbone + heads)
│           ├── network.py   # Optimizer & scheduler wrapper
│           ├── dataset.py   # Dataset loader (.pkl files)
│           ├── trainer.py   # Full training loop
│           ├── losses.py    # CrossEntropy & MSE losses
│           ├── metrics.py   # Accuracy, IoU, MAE metrics
│           ├── callbacks.py
│           ├── checkpoint.py
│           ├── logger.py
│           └── progbar.py
├── ViT/
│   ├── train.ipynb          # Training entry-point for the ViT model
│   └── Network/src/         # Mirrors CNN structure with ViT backbone
├── ViM/
│   ├── train.ipynb          # Training entry-point for the ViM model
│   └── Network/src/         # Mirrors CNN structure with ViM backbone
├── Dataset/
│   └── Real/
│       ├── train/
│       │   ├── no_def/      # No deformation
│       │   ├── normal/      # Normal fault
│       │   ├── strike-slip/ # Strike-slip fault
│       │   └── thrust/      # Thrust fault
│       └── valid/
│           └── ...
├── CheckPoints/             # Auto-saved checkpoints (by timestamp)
│   ├── CNN/
│   ├── ViT/
│   └── ViM/
├── test.ipynb               # Inference / evaluation notebook
└── tsne.ipynb               # t-SNE feature visualisation notebook
```

---

## Models

All three models share the same dual-head design:

| Model | Backbones supported |
|-------|---------------------|
| **CNN** | `resnet18`, `resnet34`, `resnet50` (via `timm`) |
| **ViT** | `vit_base_patch16_224`, `vit_base_patch8_224`, `vit_large_patch32_224` (via `timm`) |
| **ViM** | `vim_small_patch16_stride16_224`, `vim_small_patch16_stride8_224`, `vim_tiny_patch16_stride16_224`, `vim_tiny_patch16_stride8_224` (via `models_mamba`) |

**Classification head** — linear layer → softmax over `num_classes` categories.  
**Localisation head** — linear layer → 4 bounding-box coordinates.

Either head can be disabled by setting the corresponding count to `0` / `None`.

---

## Dataset format

Each sample is stored as a BZ2-compressed pickle file (`.pkl`) with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `wrapped` | `np.ndarray` | Wrapped interferogram |
| `unwrapped` | `np.ndarray` | Unwrapped interferogram |
| `label` | `int` | Class index |
| `loc` | `np.ndarray` | Bounding-box `[x_min, x_max, y_min, y_max]` (normalised) |

Samples are organised by class subfolder:

```
Dataset/Real/{train|valid}/{class_name}/*.pkl
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mei-abdallah/DeformationInterpretation.git
cd DeformationInterpretation

# Install dependencies
pip install torch torchvision timm numpy
# For ViM models, install the models_mamba package separately
```

---

## Training

Open the `train.ipynb` notebook inside the relevant model folder (`CNN/`, `ViT/`, or `ViM/`) and run all cells, **or** use the `Trainer` class directly:

```python
from Network import Trainer

trainer = Trainer(
    datadir      = '../Dataset/Real',   # path to dataset root
    backbone     = 'resnet18',          # backbone name
    mode         = 'unwrapped',         # 'wrapped' | 'unwrapped'
    num_classes  = 4,                   # set to None to disable classification
    num_locations= 4,                   # set to None to disable localisation
    color        = 'color',             # 'gray' | 'color'
    freeze       = False,               # freeze backbone weights
    pretrained   = True,                # ImageNet pre-training
    lr           = 2e-4,
    batch_size   = 32,
    seed         = 42,
)

trainer.fit(epochs=50, lambdaCLSQuality=2.0, lambdaPOSQuality=1.0)
```

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `'resnet18'` | Feature extractor |
| `mode` | `'unwrapped'` | Interferogram type |
| `num_classes` | `3` | Number of fault classes (`None` disables head) |
| `num_locations` | `4` | Bounding-box coordinates (`None` disables head) |
| `color` | `'color'` | Input channels: 1 (gray) or 3 (color) |
| `freeze` | `False` | Freeze backbone during training |
| `pretrained` | `True` | Use ImageNet pre-trained weights |
| `lr` | `2e-4` | Adam learning rate |
| `betas` | `(0.5, 0.999)` | Adam β coefficients |
| `batch_size` | `32` | Per-GPU batch size |
| `lambdaCLSQuality` | `2.0` | Classification loss weight |
| `lambdaPOSQuality` | `1.0` | Localisation loss weight |

### Training details

- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (factor=0.1, patience=5)
- **Classification loss**: Cross-Entropy
- **Localisation loss**: MSE
- **Multi-GPU**: automatic `DataParallel` when multiple GPUs are detected

---

## Checkpoints & logs

Each training run creates a timestamped subdirectory under `CheckPoints/{Model}/`:

```
CheckPoints/CNN/2024_09_24_14_18_31/
├── CNN.pth                        # Best model weights
├── CNN_2024_09_24_14_18_31.csv    # Per-epoch metric log
├── CNN_2024_09_24_14_18_31.txt    # Hyperparameter record
└── train_2024_09_24_14_18_31.ipynb
```

---

## Evaluation & visualisation

- **`test.ipynb`** — loads a saved checkpoint and evaluates on new data.
- **`tsne.ipynb`** — extracts backbone features and projects them with t-SNE to visualise class separability.

---

## Desktop Application

![App](https://user-images.githubusercontent.com/75666946/235572851-4ae92c0e-f8c5-454c-bcff-11eaa37bf6ed.png)

A companion desktop application is provided for interactive interpretation of coseismic deformation results.

---

## License

This project is released under the [MIT License](LICENSE).
