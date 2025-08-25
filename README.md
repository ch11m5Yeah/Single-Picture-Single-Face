# Single Picture Single Face

## Introduction

This project explores a lightweight approach to **single-face detection from a single image**.  
The model is designed for embedded deployment, featuring a compact backbone based on **depthwise separable convolutions** and a simplified bottleneck structure, with a **YOLO-style regression head** for bounding box prediction.

The model is not meant to outperform large-scale detectors, but to provide a practical and efficient solution for **real-time face detection on MCUs**.

## Features

- Input size: **60 × 80 pixels**  
- Output channels: **33**  
  - Channel 1: confidence score (trained with **Focal Loss**)  
  - Channels 2–33: bounding box distances (top, left, bottom, right) for 4×8 grid, normalized  
- Lightweight and efficient  
- Embedded-friendly, optimized for **RA8D1** MCU  
- Inference speed: ~8–10 ms per frame  
- Quantization support  

## Dataset

- Uses **LabelMe** annotation format  
- **Requirement:** Each annotation JSON file **must include the original image**  
- To train on your own dataset, modify the files in the `dataset` folder  

## Training

- Hyperparameters can be adjusted in `train.py`  
- `spsf_quant` controls quantization; adjust as needed  
- Training scripts require JSON files to contain both labels and images  

## Deployment

- Provides **C files** generated via **Renesas eAI tools**  
- Inference accelerated using **ARM CMSIS-NN**  
- RA8D1 performance: ~8–10 ms per frame  
- Decoding predictions can be referenced in the provided Python code or C implementation  

## Limitations

- Performance may degrade for very large faces  

## Notes

- This is a learning-stage project, meant for experimentation and embedded deployment practice.
