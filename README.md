# NPU/GPU Passthrough on Qualcomm® Hexagon™-Adreno™

**Version:** 1.0
**Release Date:** Aug 2025  
**Copyright:** © 2025 Advantech Corporation. All rights reserved.

## Overview

The `NPU/GPU Passthrough on Qualcomm® Hexagon™-Adreno™` container image provides a comprehensive environment for building and deploying AI applications on Qualcomm® DSP/NPU & GPU hardware. This container features full hardware acceleration support, optimized AI frameworks, and industrial-grade reliability. With this container, developers can quickly prototype and deploy AI use cases such as computer vision, generative AI, and large language models (LLMs) without the burden of solving time-consuming dependency issues or manually setting up complex toolchains. All required runtimes, libraries, and drivers are pre-configured, ensuring seamless integration with Qualcomm’s AI acceleration stack.

## Key Features

- **Complete AI Framework Stack:** Pre-integrated runtimes including QNN SDK (QNN, SNPE) and LiteRT for seamless execution of a wide variety of model formats (.dlc, .tflite, .so). Developers can deploy models without worrying about low-level compatibility issues.

- **Edge AI Capabilities:** Optimized support for computer vision leveraging Qualcomm’s GPU/NPU acceleration.

- **LLM & Generative AI Ready:** Container includes dependencies to run transformer-based models, enabling use cases such as chatbots, summarization, multimodal reasoning, and on-device generative AI applications.

- **Hardware Acceleration:** Direct passthrough access to NPU/GPU hardware ensures high-performance and low-latency inference with minimal power consumption.

- **Preconfigured Environment:** Eliminates time-consuming setup by bundling drivers, toolchains, and AI libraries, so developers can focus directly on building applications.

- **Rapid Prototyping & Deployment:** Ideal for quickly testing AI models, validating PoCs, and deployment without rebuilding from scratch.


## Hardware Specifications

| Component       | Specification      |
|-----------------|--------------------|
| Target Hardware | [Advantech AOM-2721](https://www.advantech.com/en/products/a9f9c02c-f4d2-4bb8-9527-51fbd402deea/aom-2721/mod_f2ab9bc8-c96e-4ced-9648-7fce99a0e24a) |
| SoC             | [Qualcomm® QCS6490](https://www.advantech.com/en/products/risc_evaluation_kit/aom-dk2721/mod_0e561ece-295c-4039-a545-68f8ded469a8)   |
| GPU             | Adreno™ 643        |
| DSP/NPU         | Hexagon™ 770       |
| Memory          | 8GB LPDDR5         |

## Operating System

This container is intended for **QCOM Robotics Reference Distro with ROS**, version **1.3-ver.1.1** OS running on QCS6490 device.

| Environment        | Operating System                                    |
|--------------------|-----------------------------------------------------|
| **Device Host**    | QCOM Robotics Reference Distro with ROS 1.3-ver.1.1 |
| **Container**      | Ubuntu 22.04 LTS 

## Software Components

| Component   | Version | Description                                                                                  |
|-------------|---------|----------------------------------------------------------------------------------------------|
| LiteRT      | 1.3.0   | Provides QNN TFLite Delegate support for GPU and DSP/NPU acceleration                            |
| [SNPE](https://docs.qualcomm.com/bundle/publicresource/topics/80-70014-15B/snpe.html)        | 2.29.0  | Qualcomm’s Snapdragon Neural Processing Engine; optimized runtime    |
| [QNN](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html)         | 2.29.0  | Qualcomm® Neural Network (QNN) runtime for executing quantized neural networks                |
| GStreamer   | 1.20.7  | Multimedia framework for building flexible audio/video pipelines                             |
| Python   | 3.10.12  | Python runtime for building applications                             |
| OpenCV    | 4.11.0 | Computer vision library for image and video processing |
| torch | 1.8.0   | Used for YOLOv8 model export only via Ultralytics export utilities                   |
| torchvision | 0.9.0  | Required alongside torch for model export (not used during inference)               |

## Supported AI Capabilities

### Vision Models

| Model                               | Format       | Note                                                                 |
|-------------------------------------|--------------|----------------------------------------------------------------------|
| YOLOv8 Detection                    | TFLite INT8  | Downloaded from Ultralytics` official source and exported to TFLite using Ultralytics Python packages |
| YOLOv8 Segmentation                 | TFLite INT8  | Downloaded from Ultralytics` official source and exported to TFLite using Ultralytics Python packages |
| YOLOv8 Pose Estimation              | TFLite INT8  | Downloaded from Ultralytics` official source and exported to TFLite using Ultralytics Python packages |
| Lightweight Face Detector           | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| FaceMap 3D Morphable Model          | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| DeepLabV3+ (MobileNet)              | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| DeepLabV3 (ResNet50)                | SNPE DLC TFLite | Converted using Qualcomm® AI Hub                                       |
| HRNet Pose Estimation (INT8)        | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| PoseNet (MobileNet V1)              | TFLite       | Converted using Qualcomm® AI Hub                                       |
| MiDaS Depth Estimation              | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| MobileNet V2 (Quantized)            | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |
| Inception V3 (SNPE DLC)             | SNPE DLC TFLite | Converted using Qualcomm® AI Hub                                       |
| YAMNet (Audio Classification)       | TFLite       | Converted using Qualcomm® AI Hub                                       |
| YOLO (Quantized)                    | TFLite INT8  | Converted using Qualcomm® AI Hub                                       |


### LLM Models

| Model                               | Format       |   Note                                                         |
|-------------------------------------|--------------|----------------------------------------------------------------|
| Phi2                                | .so          | Converted using Qualcomm's LLM Notebook for Phi-2              |
| Tinyllama                           | .so          | Converted using Qualcomm's LLM Notebook for Tinyllama          |
| Meta Llama 3.2 1B                   | .so          | Converted using Qualcomm's LLM Notebook for Meta Llama 3.2 1B  |

> **Note:** The above tables highlight a subset of commonly used models validated for this environment. Other transformer-based or vision models may also be supported depending on runtime compatibility and hardware resources. For the most detailed and updated list of supported models and runtimes, please refer to the Qualcomm's official [AI Hub](https://aihub.qualcomm.com/models).

## Supported AI Model Formats

| Runtime | Format  | Compatible Versions | 
|---------|---------|---------------------|
| QNN     | .so     |       2.29.0        |
| SNPE    | .dlc    |       2.29.0        |
| LiteRT  | .tflite |       1.3.0         | 

## Hardware Acceleration Support

| Accelerator | Support Level | Compatible Libraries |
|-------------|---------------|----------------------|
| GPU         |  FP32         | QNN, SNPE, LiteRT    |             
| DSP/NPU     |  INT8         | QNN, SNPE, LiteRT    |             


### Precision Support

| Precision  | Support Level | Notes |
|------------|---------------|-------|
| FP32       | CPU, GPU      | Baseline precision, highest accuracy  |
| INT8       | CPU, DSP/NPU  | Faster inference time, lower accuracy |

## Repository Structure
```
NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno/
├── .env                                    # Environment configuration
├── data                                    # Readme Data (images, gifs)
├── windows-git-setup.md                    # Steps to fix LF/CRLF issues on windows while copying to device
├── README.md                               # Overview and quick start steps
├── build.sh                                # Build script
├── docker-compose.yml                      # Docker Compose setup
└── wise-bench.sh                           # Script to verify acceleration and software stack inside container
```

## Quick Start Guide

### Clone the Repository (on your development machine)

> **Note for Windows Users:**  
> If you are using **Linux**, no changes are needed — LF line endings are used by default.  
> If you are on **Windows**, please follow the steps in [Windows Git Line Ending Setup](./windows-git-setup.md) before cloning to ensure scripts and configuration files work correctly on Device.

```bash
git clone https://github.com/Advantech-EdgeSync-Containers/NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno.git
cd NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno
```

### Transfer the `NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno` folder to QCS6490 device

If you cloned the repo on a **separate development machine**, use `scp` to transfer only the relevant folder:

```bash
# From your development machine (Ubuntu or Windows PowerShell if SCP is installed)
scp -r .\NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno\ <username>@<qcs6490-ip>:/home/<username>/
```

Replace:

* `<username>` – Login username on the QCS6490 board (e.g., `root`)
* `<qcs6490-ip>` – IP address of the QCS6490 board (e.g., `192.168.1.42`)

This will copy the folder to `/home/<username>/NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno/`.

Then SSH into the Qualcomm® device:

```bash
ssh <username>@<qcs6490-ip>
cd ~/NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno
```

### Installation

```bash
# Make the build script executable
chmod +x build.sh

# Launch the container
./build.sh
```
### AI Accelerator and Software Stack Verification (Optional)
```bash
# Verify AI Accelerator and Software Stack Inside Docker Container
cd /workspace
chmod +x wise-bench.sh
./wise-bench.sh
```

![qualcomm-cv-wise-bench.png](%2Fdata%2Fimages%2Fqualcomm-cv-wise-bench.png)

Wise-bench logs are saved in the `wise-bench.log` file under `/workspace`

## Model Optimization Workflows

For optimal performance, follow these recommended model conversion paths:

### PyTorch Models

```
PyTorch → ONNX → TensorFlow → LiteRT
```

### TensorFlow Models
```
TensorFlow → SavedModel → QNN(.cpp, .bin) → QNN Model Library(.so)
```

```
TensorFlow → SavedModel → SNPE(.dlc)
```

```
TensorFlow → LiteRT
```


## Best Practices

### Precision Selection
- **Quantize to INT8 for DSP acceleration:** Qualcomm’s DSP/HTP cores are optimized for INT8 workloads, providing the best trade-off between performance, accuracy, and power efficiency. Wherever possible, convert models to INT8 precision using calibration or quantization-aware training.  
- **Fallback to FP16/FP32 when necessary:** Use FP16 or FP32 precision only if INT8 quantization causes unacceptable accuracy drops. GPU execution is generally better suited for higher-precision workloads.  
- **Validate accuracy after quantization:** Always benchmark quantized models against baseline FP32 models to ensure performance gains do not compromise application requirements.

### Model Optimization
- **Use smaller backbones:** Lightweight backbones such as MobileNet, EfficientNet-Lite, or quantized YOLO variants run significantly faster on edge hardware compared to heavier models.  
- **Leverage pre-optimized models:** Start with models from Qualcomm® AI Hub or Ultralytics exports, as these are already tested for compatibility and performance.  
- **Prune and compress where possible:** Model pruning and weight clustering can reduce memory footprint and latency.

### Deployment Practices
- **Pin runtime versions:** Use validated versions of QNN, SNPE, and LiteRT that match your container to avoid runtime incompatibility.  
- **Batch size considerations:** For real-time applications (vision, speech), keep batch sizes low (often batch = 1) to minimize latency.  
- **Benchmark on-device:** Always profile models directly on the target device (QCS6490) rather than relying solely on desktop benchmarks.

### Resource Management
- **Balance compute units:** Map workloads to NPU/GPU for efficiency, but use CPU as fallbacks when certain operators are unsupported.  
- **Optimize memory usage:** Use INT8 models to fit within on-device memory limits, and offload non-critical workloads to host CPU if needed.  
- **Monitor thermal and power constraints:** Prolonged heavy inference may cause throttling; tune workloads for sustainable deployment in edge environments.

### Troubleshooting Tips
- **Operator not supported:** If certain layers/operators are not supported by QNN or SNPE, try re-exporting the model with supported ops, or use fallback execution on GPU/CPU.  
- **Conversion errors (TFLite/SNPE/QNN):** Ensure you are using the recommended exporter tools (Ultralytics for YOLO → TFLite, AI Hub for conversions). Mismatched versions often cause failures.  
- **Accuracy drop after quantization:** Re-run quantization with a larger calibration dataset, or explore quantization-aware training for sensitive models.  
- **Runtime crashes or missing libraries:** Verify that your container has all required runtime drivers (NPU/GPU libraries, SNPE/QNN SDK versions). Pinning versions avoids mismatches.  
- **Performance lower than expected:** Check if inference is actually running on DSP/NPU; fallback to CPU/GPU can occur silently if the runtime doesn’t support certain ops.  


## Known Limitations

- **LiteRT Runtime:** DSP/NPU acceleration coverage is limited for complex or custom operators. Models containing unsupported layers may partially fall back to CPU/GPU, which can reduce performance.  
- **Operator Coverage Gaps:** Some advanced layers (e.g., attention mechanisms or custom TensorFlow/PyTorch ops) may not be directly supported by QNN or SNPE runtimes. Such models may require modifications, re-training, or partial execution on CPU/GPU.  
- **Quantization Trade-offs:** INT8 quantization is strongly recommended for DSP/NPU acceleration, but aggressive quantization may lead to accuracy degradation for certain tasks.  
- **Resource Constraints:** Larger transformer-based models (LLMs beyond ~1–3B parameters) may not fit into memory or may run at impractical speeds on QCS6490. Smaller optimized models (Phi-2, Llama 3.2 1B, distilled transformers) are recommended.  
- **Version Compatibility:** Model conversion/export pipelines (e.g., Ultralytics → TFLite, AI Hub conversions) may break if mismatched with runtime SDK versions (QNN, SNPE). Always verify compatibility with official Qualcomm® release notes.


## Possible Use Cases

1. **Smart Surveillance & Security**  
   - Real-time object detection and person tracking using YOLO-based models.  
   - Intrusion detection, face recognition, and abnormal behavior monitoring on edge devices without cloud dependency.  

2. **Industrial Automation & Robotics**  
   - Defect detection in manufacturing lines with computer vision.  
   - Gesture or pose estimation for human–robot collaboration.  
   - Autonomous navigation and obstacle avoidance for robots and drones.  

3. **Healthcare & Wellness**  
   - Contactless vital sign monitoring using vision models.  
   - Fall detection and activity recognition for elderly care.  
   - Medical imaging assistance with lightweight segmentation models.  

4. **Retail & Smart Spaces**  
   - Customer flow analysis, heatmap generation, and people counting.  
   - Shelf stock monitoring and automated checkout solutions.  
   - Emotion detection for personalized customer experiences.  

5. **Transportation & Mobility**  
   - Driver monitoring (drowsiness, distraction detection).  
   - Traffic analysis and smart signaling.  
   - Vehicle and license plate recognition at the edge.  

6. **Speech & Natural Language Applications**  
   - On-device assistants powered by lightweight LLMs (e.g., Phi-2, Llama 3.2 1B).  
   - Multilingual translation, summarization, and intent detection.  
   - Edge speech-to-text and conversational AI for low-latency use. 

## Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)**: For the YOLOv8 framework that powers this toolkit. 
  ```bash
  # Required specific versions:
  python3 -m pip install ultralytics==8.3.176 --no-deps
  python3 -m pip install ultralytics-thop==2.0.0 --no-deps
  ```
  
© 2025 Advantech Corporation. All rights reserved.