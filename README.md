
## You Only Once for Panoptic Perception

YOLOP (You Only Look Once for Panoptic driving Perception) is a real-time, multi-task neural network for autonomous driving that performs traffic object detection, drivable area segmentation, and lane detection simultaneously.

<img width="838" height="488" alt="{49260BA3-D870-4E8B-85C9-1BC0D28D24FF}" src="https://github.com/user-attachments/assets/5c00c7c3-cd57-472a-94fa-bbba9f1fecdf" />

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolop-you-only-look-once-for-panoptic-driving/traffic-object-detection-on-bdd100k)](https://paperswithcode.com/sota/traffic-object-detection-on-bdd100k?p=yolop-you-only-look-once-for-panoptic-driving)

---

## Quick Start

### 1. Installation

This codebase has been developed with python version 3.7, PyTorch 1.7+ and torchvision 0.8+:

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### 2. Data Preparation

Download the BDD100K dataset and annotations:
- [Images](https://bdd-data.berkeley.edu/)
- [Detection Annotations](https://drive.google.com/file/d/1Ge-R8NTxG1eqd4zbryFo-1Uonuh0Nxyl/view?usp=sharing)
- [Drivable Area Annotations](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view?usp=sharing)
- [Lane Line Annotations](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view?usp=sharing)

Organize your dataset as follows and update the paths in `./lib/config/default.py` (specifically `_C.DATASET.DATAROOT` and related fields):

```
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

### 3. Training

Train the model with default configuration:

```bash
python tools/train.py
```

For multi-GPU training:
```bash
python -m torch.distributed.launch --nproc_per_node=N tools/train.py
```

### 4. Inference / Demo

Run inference on images or videos using the standard demo script:

```bash
# Run on a folder of images (ensure the path exists or is created)
python tools/demo.py --source inference/images --weights weights/End-to-end.pth

# Run on webcam (default 0)
python tools/demo.py --source 0
```

#### Side-by-Side Comparison Demo
For a better visualization that shows the original video and the processed perception result side-by-side, use the `src/demo_side_by_side.py` script:

```bash
python src/demo_side_by_side.py --source path/to/video.mp4 --weights weights/End-to-end.pth
```

### 5. Evaluation

Evaluate the model on the validation set:

```bash
python tools/test.py --weights weights/End-to-end.pth
```

### 6. Running Tests

This repository includes unit tests to verify the core utility functions:

```bash
python -m unittest discover tests
```

---

## Project Structure

```
├─lib/                # Core library
│ ├─config           # Configuration files (Update default.py for your paths)
│ ├─core             # Core training/eval capabilities
│ ├─dataset          # Dataset loaders (BDD100K)
│ ├─models           # YOLOP model definition
│ ├─utils            # Utilities (logging, plotting, etc.)
├─tools/              # Main execution scripts
│ ├─demo.py          # Standard inference script
│ ├─test.py          # Evaluation script
│ ├─train.py         # Training script
├─src/                # Additional tools
│ ├─demo_side_by_side.py  # Side-by-side visualization demo
├─tests/              # Unit tests for the codebase
├─weights/            # Pre-trained weights (.pth and .onnx formats)
```

---

## Performance & Results

### Traffic Object Detection

| Model          | Recall(%) | mAP50(%) | Speed(fps) |
| -------------- | --------- | -------- | ---------- |
| `YOLOP(ours)`  | 89.2      | 76.5     | 41         |

### Drivable Area Segmentation

| Model         | mIOU(%) | Speed(fps) |
| ------------- | ------- | ---------- |
| `YOLOP(ours)` | 91.5    | 41         |

### Lane Detection

| Model         | mIOU(%) | IOU(%) |
| ------------- | ------- | ------ |
| `YOLOP(ours)` | 70.50   | 26.20  |

For detailed ablation studies and comparisons with other models, please refer to our [Paper](https://arxiv.org/abs/2108.11250).

---

## Deployment & Export

### TensorRT Deployment
Check the `toolkits/deploy` folder for C++ implementation on Jetson TX2.

### ONNX Weights
We provide pre-trained ONNX weights in the `weights/` directory for different resolutions:
- `yolop-320-320.onnx`
- `yolop-640-640.onnx`
- `yolop-1280-1280.onnx`

---

## 📝 Citation

If you find our paper and code useful for your research, please consider giving a star :star: and citation :pencil: :

```BibTeX
@article{wu2022yolop,
  title={Yolop: You only look once for panoptic driving perception},
  author={Wu, Dong and Liao, Man-Wen and Zhang, Wei-Tian and Wang, Xing-Gang and Bai, Xiang and Cheng, Wen-Qing and Liu, Wen-Yu},
  journal={Machine Intelligence Research},
  pages={1--13},
  year={2022},
  publisher={Springer}
}
```
