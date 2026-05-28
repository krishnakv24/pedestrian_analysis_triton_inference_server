# Model Service — Triton Inference Server

Production-ready inference service for the pedestrian analysis CV pipeline.
Serves YOLOv8 TensorRT models via **NVIDIA Triton Inference Server** over **gRPC**.
Designed to scale — LLM agent brain (Qwen / LLaMA) can be added alongside via vLLM.

---

## Repository Structure

```
pedestrian_analysis_triton_inference_server/
├── model_repository/
│   ├── person_detection/          # YOLOv8n — person detection
│   │   ├── config.pbtxt
│   │   └── 1/model.plan           # TensorRT FP16 engine
│   ├── face_detection/            # YOLOv8l-face — face detection
│   │   ├── config.pbtxt
│   │   └── 1/model.plan
│   └── face_reid/                 # YOLOv8n-face — face re-identification
│       ├── config.pbtxt
│       └── 1/model.plan
├── input/                         # Sample test images
├── output/                        # Inference result images
├── Dockerfile                     # Triton server image definition
├── docker-compose.yml             # Service orchestration
├── test_client.py                 # Full pipeline gRPC client
├── host_machine_requirements.txt  # Host-side Python dependencies
└── README.md
```

---

## Models

| Model Name | Purpose | Input | Output Shape |
|---|---|---|---|
| `person_detection` | Detect persons in frame | 3 × 640 × 640 | (batch, 84, 8400) |
| `face_detection` | Detect faces per person crop | 3 × 640 × 640 | (batch, 5, 8400) |
| `face_reid` | Re-identify faces | 3 × 640 × 640 | (batch, 5, 8400) |

> **Engine settings:** FP16 · Batch 1–4 · Input 640×640 · TensorRT 10.3
> 
> Model files are named `model.plan` — required naming convention for Triton's TensorRT backend.

---

## Inference Pipeline

```
Input Image
    │
    ▼
[Person Detection]  ── YOLOv8n       →  person bounding boxes
    │
    ▼  (crop each person region)
[Face Detection]    ── YOLOv8l-face  →  face bounding boxes
    │
    ▼  (crop each face region)
[Face ReID]         ── YOLOv8n-face  →  face identity features
    │
    ▼
Annotated Output Image
```

---

## Server Requirements

- NVIDIA GPU (tested: RTX 4050 6GB)
- CUDA driver 12.x
- Docker
- NVIDIA Container Toolkit

### Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Build the Docker Image

### Step 1 — Clone the repository

```bash
git clone https://github.com/krishnakv24/pedestrian_analysis_triton_inference_server.git
cd pedestrian_analysis_triton_inference_server
```

### Step 2 — Build the image

```bash
docker build -t triton-service:latest .
```

**What the Dockerfile does:**
- Starts from `nvcr.io/nvidia/tritonserver:24.08-py3` (NVIDIA official Triton image)
- Installs `tritonclient[grpc]` and `numpy` for the gRPC client
- Exposes ports `8001` (gRPC) and `8002` (metrics)
- Sets `tritonserver` as the default command with the model repository mounted at `/models`

### Step 3 — Verify the image

```bash
docker images triton-service
```

Expected:

```
REPOSITORY       TAG       IMAGE ID       CREATED        SIZE
triton-service   latest    xxxxxxxxxxxx   X minutes ago  ~9GB
```

---

## Start the Service

```bash
cd pedestrian_analysis_triton_inference_server
docker compose up -d
```

Check all models loaded successfully:

```bash
docker logs triton_vision
```

Expected:

```
successfully loaded 'person_detection'
successfully loaded 'face_detection'
successfully loaded 'face_reid'
Started GRPCInferenceService at 0.0.0.0:8001
Started Metrics Service at 0.0.0.0:8002
```

---

## Host Machine Setup

Install Python dependencies on the client machine:

```bash
pip install -r host_machine_requirements.txt
```

---

## Run the Pipeline Test

Place input images (`.png` / `.jpg`) in the `input/` folder, then:

```bash
python3 test_client.py
```

### Sample Output

```
============================================================
  Triton Inference Pipeline
  Person Detection → Face Detection → Face ReID
============================================================
  Server  : localhost:8001
  GPU     : GPU   0%  VRAM 819/6141 MB
  Input   : ./input
  Output  : ./output
============================================================

  Image: people_with_phones.png  (1238x750)
  [Person Detection]   4 persons  | 43.5 ms | GPU   3%  VRAM 819/6141 MB
  [Face  Detection]    4 faces    | 23.4 ms | GPU  85%  VRAM 819/6141 MB
  [Face  ReID    ]     4 faces    | 13.0 ms | GPU  85%  VRAM 819/6141 MB
  Saved → output/people_with_phones.png

============================================================
  Done — 4 images processed
  Final GPU: GPU  34%  VRAM 819/6141 MB
============================================================
```

Annotated output images are saved to `output/` with:
- **Green boxes** — detected persons
- **Red boxes** — detected faces
- **Purple labels** — face IDs (ID-0, ID-1 ...)

---

## gRPC Endpoints

| Port | Protocol | Purpose |
|---|---|---|
| `8001` | gRPC | Model inference |
| `8002` | HTTP | Prometheus metrics |

### Custom Inference (Python)

```python
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")

image = np.random.rand(1, 3, 640, 640).astype(np.float32)
inp = grpcclient.InferInput("images", image.shape, "FP32")
inp.set_data_from_numpy(image)
out = grpcclient.InferRequestedOutput("output0")

# Choose: person_detection | face_detection | face_reid
result = client.infer(model_name="person_detection", inputs=[inp], outputs=[out])
detections = result.as_numpy("output0")  # shape: (1, 84, 8400)
```

---

## Stop the Service

```bash
docker compose down
```

---

## Future — LLM Agent Brain

The `docker-compose.yml` includes a pre-configured `vllm` service block (commented out) for adding **Qwen** or **LLaMA** as the agent brain.

To enable:
1. Place safetensor model in `./llm_models/`
2. Uncomment the `vllm` block in `docker-compose.yml`
3. Run `docker compose up -d vllm`

The vLLM service exposes an **OpenAI-compatible API** on port `8000`.

---

## Author

**krishnaprasad kv** — [krishnakv24](https://github.com/krishnakv24)
