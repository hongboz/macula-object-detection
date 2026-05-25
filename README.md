# Macula & Optic Disc Detection — YOLO-NAS Transfer Learning

A transfer-learning project that fine-tunes **YOLO-NAS-m** (pre-trained on COCO) to detect two anatomical structures in retinal fundus images: the **macula** and the **optic disc**. Includes a training notebook, an ONNX export script, and a Hugging Face Spaces (Gradio) inference demo.

## About this project

This project was an experiment to see if the YOLO series of object detection models is able to detect small objects (10-50 pixels) in a grayscale image. If it is, then it should also be able to
detect features in Oil and Gas MFL pipeline data gathered by NDT companies as both data can be
represented as single channel (grayscale) images.

## Demo

<img width="2093" height="1297" alt="image" src="https://github.com/user-attachments/assets/efe71603-3305-4383-a27a-6e07f6caea2c" />


> No live Space is hosted — but the demo is one click to deploy yourself on the free Hugging Face CPU tier.

### Deploy your own Hugging Face Space

1. **Train and export.** Follow [Reproduce the training](#reproduce-the-training) below to get a `ckpt_best.pth`, then convert it:
   ```bash
   python export_model.py --checkpoint checkpoints/yolo_nas_m/RUN_<your-run>/ckpt_best.pth
   ```
   The resulting `.onnx` is written next to the `.pth`. Copy it into `huggingface_spaces/` and rename it to `ckpt_best.onnx` (the filename `app.py` expects).
2. **Create a Space.** Sign in at [huggingface.co/spaces](https://huggingface.co/spaces) → *Create new Space* → pick **Gradio** as the SDK → choose the free CPU hardware tier.
3. **Push the demo.** Clone your new (empty) Space repo, copy the contents of `huggingface_spaces/` into it (including your `.onnx`), commit, and push. The Space builds and goes live in ~2 minutes.

> `huggingface_spaces/README.md` already contains the SDK front-matter HF Spaces requires — no edits needed.

## Results

| Model       | Image size | Best validation mAP@0.50:0.95 |
| ----------- | ---------- | ----------------------------- |
| YOLO-NAS-m  | 640×640    | **0.527**                     |

## Dataset

This repo does **not** include the training data. Download it from: https://www.kaggle.com/datasets/hongbozhang/grayscale-retinal-fundus-images

After downloading, arrange the contents at `./datasets/eyes/` so the structure looks like:

```
datasets/eyes/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
└── classes.txt
```

The notebook expects YOLO-format labels (one `.txt` per image, `class cx cy w h` normalized). Classes: `macula`, `optic-disc`.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

> Python 3.10 is recommended — it's the version `super-gradients==3.7.1` is most stable on.

## Reproduce the training

1. Set up the dataset as described above.
2. Open `detection_transfer_learning.ipynb` in Jupyter or VS Code.
3. Run all cells top-to-bottom. Checkpoints land in `./checkpoints/yolo_nas_m/RUN_<timestamp>/`.

> Training takes ~80 min on a single mid-range GPU for 25 epochs over 500 training images.

## Run inference

After training, you have a `ckpt_best.pth` checkpoint. The included demo runs on the ONNX export so it's CPU-friendly and portable.

Export your `.pth` to ONNX:
```bash
python export_model.py --checkpoint checkpoints/yolo_nas_m/RUN_<your-run>/ckpt_best.pth
```

Move the resulting `.onnx` into `huggingface_spaces/` (rename to `ckpt_best.onnx`) and run locally:
```bash
cd huggingface_spaces
pip install -r requirements.txt
python app.py    # opens a Gradio UI in your browser
```

To deploy the same demo publicly, follow the three steps in [Demo](#demo) above.

## Project structure

```
.
├── detection_transfer_learning.ipynb   # main training notebook
├── export_model.py                     # .pth → .onnx converter
├── huggingface_spaces/                 # Gradio + ONNX inference demo
│   ├── app.py
│   ├── requirements.txt
│   ├── packages.txt
│   └── README.md                       # HF Spaces metadata
├── requirements.txt                    # notebook dependencies
└── .gitignore
```

## Pretrained weights

Trained checkpoints (`.pth`, ~650 MB) and exported ONNX files (~122 MB) exceed GitHub's 100 MB per-file limit and are not committed to the repo. Download them from the **[Releases page](../../releases)** (or train your own via the notebook).

## License

MIT — see `LICENSE`.

> Note: YOLO-NAS pretrained weights carry their own license terms — see the [super-gradients YOLO-NAS license](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).
