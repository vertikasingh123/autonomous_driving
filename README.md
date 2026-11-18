# Urban Scene Segmentation for Autonomous Driving

[![CI](https://github.com/vertikasingh123/autonomous_driving/actions/workflows/ci.yml/badge.svg)](https://github.com/vertikasingh123/autonomous_driving/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vertikasingh123/autonomous_driving/blob/main/notebooks/train_colab.ipynb)

Pixel-level understanding of street scenes: every pixel of a dashcam image is
classified as road, sidewalk, car, pedestrian, traffic sign and so on — 19
classes in total. The model is trained and evaluated on the
[Cityscapes](https://www.cityscapes-dataset.com/) benchmark using the standard
protocol, so its numbers are directly comparable with published results.

<!-- After training, generate this image with `python -m cityseg.visualize`
     (see "Qualitative results" below), commit it, and uncomment the next line.
![Input, ground truth and prediction](assets/qualitative.png)
-->

## Highlights

- **Standard Cityscapes protocol** — 19 training classes, void pixels ignored
  in both loss and metrics, mIoU computed from a dataset-level confusion matrix
  against full-resolution (1024×2048) labels.
- **Modern training recipe** — combined cross-entropy + Dice loss, AdamW with
  warm-up and polynomial decay, mixed precision, random-scale (0.5–2×) crops,
  flips and colour jitter.
- **Swappable architectures** — U-Net/ResNet-34 baseline and
  DeepLabV3+/ResNet-50, selected by a YAML config; any
  [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
  architecture or encoder works without code changes.
- **Complete tooling** — command-line training (resumable), evaluation,
  batch prediction, speed benchmark, qualitative figures and an interactive
  Gradio demo.
- **Tested** — unit tests for labels, augmentation and metrics, plus an
  end-to-end train → evaluate → predict test on synthetic data, run in CI.

## Results

Cityscapes `val` (500 images), single scale, evaluated at full resolution.

<!-- TODO: fill in from the `results.md` written by `python -m cityseg.evaluate`
     and the output of `python -m cityseg.benchmark`. -->

| Model | Encoder | Params | mIoU | mIoU (+flip) | Pixel acc. | FPS (1024×2048) |
|---|---|---|---|---|---|---|
| U-Net | ResNet-34 | 24.4 M | – | – | – | – |
| DeepLabV3+ | ResNet-50 | 26.7 M | – | – | – | – |

## Quickstart

```bash
git clone https://github.com/vertikasingh123/autonomous_driving.git
cd autonomous_driving
pip install -e .
```

The fastest way to train is the Colab notebook
([notebooks/train_colab.ipynb](notebooks/train_colab.ipynb)), which downloads
the data and runs every step below on a free GPU.

### Data

Register at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/),
download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`, and
unzip both into one folder:

```
data/cityscapes/
├── leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
└── gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png
```

### Train

```bash
python -m cityseg.train --config configs/unet_r34.yaml --data-root data/cityscapes
```

Checkpoints (`best.ckpt` by validation mIoU, and `last.ckpt`), the resolved
config, and CSV + TensorBoard logs are written to `outputs/unet_r34/`. Add
`--resume auto` to continue an interrupted run.

### Evaluate

```bash
python -m cityseg.evaluate --checkpoint outputs/unet_r34/checkpoints/best.ckpt \
    --data-root data/cityscapes            # add --flip for test-time augmentation
```

Prints mIoU, pixel accuracy and per-class IoU, and saves them as
`metrics.json` and `results.md`.

### Predict

```bash
python -m cityseg.predict --checkpoint outputs/unet_r34/checkpoints/best.ckpt \
    --input path/to/images --output results --save color overlay labelids
```

Works on a single image or a folder (recursively). Outputs keep the input's
file names, folder structure and resolution. `labelids` writes masks in the
Cityscapes submission format.

### Benchmark, figures and demo

```bash
python -m cityseg.benchmark --checkpoint outputs/unet_r34/checkpoints/best.ckpt
python -m cityseg.visualize --checkpoint outputs/unet_r34/checkpoints/best.ckpt \
    --data-root data/cityscapes --output assets/qualitative.png

pip install -e ".[demo]"
python app.py --checkpoint outputs/unet_r34/checkpoints/best.ckpt
```

## Method

| | |
|---|---|
| **Task** | 19-class semantic segmentation; the other 15 Cityscapes labels are void and ignored |
| **Model** | Encoder–decoder with an ImageNet-pretrained encoder (U-Net/ResNet-34 baseline) |
| **Loss** | Cross-entropy + 0.5 × Dice, both ignoring void pixels |
| **Optimiser** | AdamW, lr 3e-4, weight decay 1e-4, linear warm-up then poly decay (power 0.9) |
| **Augmentation** | Random scale 0.5–2×, 512×1024 crop, horizontal flip, colour jitter |
| **Training** | 60 epochs, batch size 8, mixed precision |
| **Evaluation** | Full 1024×2048 images; logits upsampled bilinearly before the argmax; optional flip TTA |

Design decisions worth noting:

- **Void pixels are ignored, not learned.** Roughly a tenth of Cityscapes
  pixels carry no evaluated label (ego vehicle, image borders, ambiguous
  regions). Treating them as a 20th class makes the model spend capacity on
  them and makes its scores incomparable with the benchmark.
- **IoU comes from one confusion matrix over the whole split.** Averaging
  per-image IoUs is undefined for images that lack a class and over-weights
  small objects in the images where they do appear.
- **Augmentation is train-only.** Validation and prediction use a
  deterministic transform, so the reported numbers are reproducible and
  predictions line up with their inputs.
- **Padding uses the ignore label.** When a random-scale crop extends beyond
  the image, the padded area is excluded from the loss rather than labelled as
  a real class.

## Project structure

```
├── configs/                 experiment configs (YAML)
├── notebooks/train_colab.ipynb
├── src/cityseg/
│   ├── labels.py            class definitions, id ↔ train-id mapping, palette
│   ├── data.py              dataset, augmentation, LightningDataModule
│   ├── model.py             LightningModule: network, loss, optimiser, schedule
│   ├── metrics.py           confusion-matrix mIoU / pixel accuracy
│   ├── inference.py         padded + flip-TTA inference shared by all tools
│   ├── train.py  evaluate.py  predict.py  benchmark.py  visualize.py
│   └── config.py            defaults + YAML merging
├── tests/                   unit and end-to-end tests
└── app.py                   Gradio demo
```

## Development

```bash
pip install -e ".[dev]"
pytest        # ~15 s on CPU, no dataset or GPU needed
ruff check .
```

## Acknowledgements

- [Cityscapes dataset](https://www.cityscapes-dataset.com/) (Cordts et al., CVPR 2016).
  The dataset is not redistributed here and is subject to its own licence.
- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
  and [Lightning](https://lightning.ai/).

This project began as a hackathon entry and was later rebuilt as a
reproducible, tested codebase.
