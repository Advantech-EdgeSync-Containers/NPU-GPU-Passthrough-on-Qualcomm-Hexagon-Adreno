Harness the performance of Qualcomm® Hexagon™ DSP/NPU with a containerized AI runtime optimized for vision and language workloads on the QCS6490 platform. This container includes pre-integrated AI frameworks—QNN, SNPE, and LiteRT—alongside quantized models such as Meta Llama 3.2 1B and YOLOv8, enabling low-latency inference directly on edge hardware. Designed for real-time AI applications, it supports direct DSP/NPU passthrough, allowing developers to deploy computer vision and lightweight LLMs without managing toolchains, drivers, or runtime configurations.
# DSP/NPU Passthrough on Qualcomm® Hexagon™

### About Advantech Container Catalog
The Advantech Container Catalog provides pre-integrated, hardware-accelerated containers purpose-built for rapid AI development at the edge. With support for platforms like the Qualcomm® QCS6490, these containers abstract away low-level system complexities—allowing developers to focus on building intelligent applications instead of managing toolchains, drivers, or runtime dependencies.

### Key benefits of the Container Catalog include:
| Feature / Benefit             | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| Hardware-Accelerated Edge AI  | DSP, NPU, and GPU passthrough for low-latency AI inference                 |
| Out-of-the-Box Deployment     | Pre-bundled runtimes, drivers, and toolchains for rapid startup            |
| CV & LLM Workloads Ready      | Optimized for computer vision and small-footprint transformer models       |
| Quantized Model Execution     | Supports INT8 execution on DSP for optimal performance/power efficiency    |
| Format Flexibility            | Supports TFLite, DLC, and .so model formats across multiple runtimes       |
| Developer-Centric Workflows   | Export, convert, and benchmark models using provided scripts and utilities |
| Open, Extendable Architecture | Integrates easily with existing ROS-based robotics and AI pipelines        |


## Container Overview

**DSP/NPU Passthrough on Qualcomm® Hexagon™** is a plug-and-play container that enables AI developers to run optimized, quantized models on **Qualcomm® QCS6490** with direct access to DSP/NPU accelerators. Built for high-performance edge inference, this container supports real-time computer vision, LLMs, and generative AI—without requiring manual setup of drivers, SDKs, or AI runtimes.

This container offers:

* **DSP/NPU passthrough** for accelerated INT8 inference using QNN, SNPE, and LiteRT runtimes
* Pre-integrated **LiteRT, SNPE, QNN SDKs** with validated versions to avoid runtime mismatches
* Support for **TFLite**, **DLC**, and **shared object (.so)** model formats
* Deployment-ready models including **YOLOv8**, **Segmentation**, **Pose Estimation**, and **Meta Llama 3.2 1B**
* Sample scripts for **model quantization, export, and benchmarking (wise-bench.sh)**
* Built-in compatibility with **Qualcomm® Robotics Reference Distro with ROS**
* Real-time computer vision pipelines via **GStreamer** and **OpenCV**
* Use-case ready for smart surveillance, robotics, on-device assistants, and more
* Quantization-aware deployment guidance and conversion pipelines included


## Use Cases

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

## Key Features

- **Complete AI Framework Stack:** Pre-integrated runtimes including QNN SDK (QNN, SNPE) and LiteRT for seamless execution of a wide variety of model formats (.dlc, .tflite, .so). Developers can deploy models without worrying about low-level compatibility issues.

- **Edge AI Capabilities:** Optimized support for computer vision leveraging Qualcomm’s DSP/GPU/NPU acceleration.

- **LLM & Generative AI Ready:** Container includes dependencies to run transformer-based models, enabling use cases such as chatbots, summarization, multimodal reasoning, and on-device generative AI applications.

- **Hardware Acceleration:** Direct passthrough access to DSP/NPU hardware ensures high-performance and low-latency inference with minimal power consumption.

- **Preconfigured Environment:** Eliminates time-consuming setup by bundling drivers, toolchains, and AI libraries, so developers can focus directly on building applications.

- **Rapid Prototyping & Deployment:** Ideal for quickly testing AI models, validating PoCs, and deployment without rebuilding from scratch.

## Host Device Prerequisites

| Component       | Specification      |
|-----------------|--------------------|
| Target Hardware | [Advantech AOM-2721](https://www.advantech.com/en/products/a9f9c02c-f4d2-4bb8-9527-51fbd402deea/aom-2721/mod_f2ab9bc8-c96e-4ced-9648-7fce99a0e24a) |
| SoC             | [Qualcomm® QCS6490](https://www.advantech.com/en/products/risc_evaluation_kit/aom-dk2721/mod_0e561ece-295c-4039-a545-68f8ded469a8)   |
| GPU             | Adreno™ 643        |
| DSP             | Hexagon™ 770       |
| Memory          | 8GB LPDDR5         |
| Host OS         | QCOM Robotics Reference Distro with ROS 1.3-ver.1.1       |


## Container Environment Overview

### Software Components on Container Image

| Component   | Version | Description                                                                                  |
|-------------|---------|----------------------------------------------------------------------------------------------|
| LiteRT      | 1.3.0   | Provides QNN TFLite Delegate support for GPU and DSP acceleration                            |
| [SNPE](https://docs.qualcomm.com/bundle/publicresource/topics/80-70014-15B/snpe.html)        | 2.29.0  | Qualcomm’s Snapdragon Neural Processing Engine; optimized runtime for Snapdragon DSP/HTP     |
| [QNN](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html)         | 2.29.0  | Qualcomm® Neural Network (QNN) runtime for executing quantized neural networks                |
| GStreamer   | 1.20.7  | Multimedia framework for building flexible audio/video pipelines                             |
| Python   | 3.10.12  | Python runtime for building applications                             |
| OpenCV    | 4.11.0 | Computer vision library for image and video processing |


### Container Quick Start Guide
For container quick start, including the docker-compose file and more, please refer to [README.](https://github.com/Advantech-EdgeSync-Containers/NPU-GPU-Passthrough-on-Qualcomm-Hexagon-Adreno/blob/main/README.md)

### Supported AI Capabilities

#### Vision Models

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

### Language Models Recommendation

| Model                               | Format       |   Note                                                         |
|-------------------------------------|--------------|----------------------------------------------------------------|
| Phi2                                | .so          | Converted using Qualcomm's LLM Notebook for Phi-2              |
| Tinyllama                           | .so          | Converted using Qualcomm's LLM Notebook for Tinyllama          |
| Meta Llama 3.2 1B                   | .so          | Converted using Qualcomm's LLM Notebook for Meta Llama 3.2 1B  |                                   |

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
| DSP         |  INT8         | QNN, SNPE, LiteRT    |   

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
- **Balance compute units:** Map workloads to DSP/NPU for efficiency, but use GPU/CPU as fallbacks when certain operators are unsupported.  
- **Optimize memory usage:** Use INT8 models to fit within on-device memory limits, and offload non-critical workloads to host CPU if needed.  
- **Monitor thermal and power constraints:** Prolonged heavy inference may cause throttling; tune workloads for sustainable deployment in edge environments.

### Troubleshooting Tips
- **Operator not supported:** If certain layers/operators are not supported by QNN or SNPE, try re-exporting the model with supported ops, or use fallback execution on GPU/CPU.  
- **Conversion errors (TFLite/SNPE/QNN):** Ensure you are using the recommended exporter tools (Ultralytics for YOLO → TFLite, AI Hub for conversions). Mismatched versions often cause failures.  
- **Accuracy drop after quantization:** Re-run quantization with a larger calibration dataset, or explore quantization-aware training for sensitive models.  
- **Runtime crashes or missing libraries:** Verify that your container has all required runtime drivers (DSP/NPU libraries, SNPE/QNN SDK versions). Pinning versions avoids mismatches.  
- **Performance lower than expected:** Check if inference is actually running on DSP/NPU; fallback to CPU/GPU can occur silently if the runtime doesn’t support certain ops.  
