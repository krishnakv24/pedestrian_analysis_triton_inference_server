import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import pynvml
import time
from pathlib import Path

TRITON_URL  = "localhost:8001"
INPUT_DIR   = Path(__file__).parent / "input"
OUTPUT_DIR  = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_THRESH = 0.3
IOU_THRESH  = 0.45
INPUT_SIZE  = 640

# ── GPU Monitor ───────────────────────────────────────────────────────────────

pynvml.nvmlInit()
_gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(_gpu)
    util = pynvml.nvmlDeviceGetUtilizationRates(_gpu)
    used = mem.used / 1024**2
    total = mem.total / 1024**2
    return f"GPU {util.gpu:3d}%  VRAM {used:.0f}/{total:.0f} MB"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_bgr):
    """BGR image → (1, 3, 640, 640) float32 tensor, return scale info."""
    h, w = img_bgr.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None], scale, (nh, nw)


# ── Post-processing ───────────────────────────────────────────────────────────

def xywh2xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)

def nms(boxes, scores, iou_thresh):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thresh]
    return keep

def compute_iou(box, others):
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    a1 = (box[2]-box[0]) * (box[3]-box[1])
    a2 = (others[:,2]-others[:,0]) * (others[:,3]-others[:,1])
    return inter / (a1 + a2 - inter + 1e-6)

def postprocess(output, orig_hw, scale, nw_nh, conf_thresh, person_only=False):
    """
    output: (1, num_classes+4, 8400)
    Returns list of [x1,y1,x2,y2,conf] in original image coords.
    """
    pred = output[0].T                          # (8400, num_classes+4)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    if person_only:
        scores = class_scores[:, 0]             # class 0 = person (COCO)
    else:
        scores = class_scores[:, 0]             # single class (face)

    mask = scores > conf_thresh
    if not mask.any():
        return []

    boxes = xywh2xyxy(boxes_xywh[mask])
    scores = scores[mask]

    # scale back to original image coords
    boxes = boxes / scale
    oh, ow = orig_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

    keep = nms(boxes, scores, IOU_THRESH)
    return [[*boxes[i].tolist(), float(scores[i])] for i in keep]


# ── Triton Inference ──────────────────────────────────────────────────────────

def infer(client, model_name, tensor):
    inp = grpcclient.InferInput("images", tensor.shape, "FP32")
    inp.set_data_from_numpy(tensor)
    out = grpcclient.InferRequestedOutput("output0")
    return client.infer(model_name=model_name, inputs=[inp], outputs=[out]).as_numpy("output0")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(client, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Cannot read {img_path.name}")
        return
    orig_h, orig_w = img.shape[:2]
    vis = img.copy()

    print(f"\n  Image: {img_path.name}  ({orig_w}x{orig_h})")

    # ── Stage 1: Person Detection ─────────────────────────────────────────────
    t0 = time.perf_counter()
    tensor, scale, nw_nh = preprocess(img)
    out = infer(client, "person_detection", tensor)
    persons = postprocess(out, (orig_h, orig_w), scale, nw_nh, CONF_THRESH, person_only=True)
    t1 = time.perf_counter()
    print(f"  [Person Detection]  {len(persons):2d} persons  | {(t1-t0)*1000:.1f} ms | {gpu_stats()}")

    for p in persons:
        x1,y1,x2,y2,conf = p
        cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(vis, f"person {conf:.2f}", (int(x1), int(y1)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ── Stage 2: Face Detection on each person crop ───────────────────────────
    all_faces = []
    for p in persons:
        x1,y1,x2,y2,_ = p
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        t0 = time.perf_counter()
        tensor, scale_f, nw_nh_f = preprocess(crop)
        out = infer(client, "face_detection", tensor)
        faces = postprocess(out, (int(y2-y1), int(x2-x1)), scale_f, nw_nh_f, CONF_THRESH)
        t1 = time.perf_counter()

        # map face coords back to original image
        for f in faces:
            f[0] += x1; f[2] += x1
            f[1] += y1; f[3] += y1
            all_faces.append(f)

    print(f"  [Face  Detection]   {len(all_faces):2d} faces   | {(t1-t0)*1000:.1f} ms | {gpu_stats()}")

    for f in all_faces:
        fx1,fy1,fx2,fy2,conf = f
        cv2.rectangle(vis, (int(fx1),int(fy1)), (int(fx2),int(fy2)), (0,0,255), 2)
        cv2.putText(vis, f"face {conf:.2f}", (int(fx1), int(fy1)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    # ── Stage 3: Face ReID on each face crop ─────────────────────────────────
    for i, f in enumerate(all_faces):
        fx1,fy1,fx2,fy2,_ = f
        face_crop = img[int(fy1):int(fy2), int(fx1):int(fx2)]
        if face_crop.size == 0:
            continue
        t0 = time.perf_counter()
        tensor, scale_r, nw_nh_r = preprocess(face_crop)
        reid_out = infer(client, "face_reid", tensor)
        t1 = time.perf_counter()
        cv2.putText(vis, f"ID-{i}", (int(fx1), int(fy2)+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

    print(f"  [Face  ReID    ]    {len(all_faces):2d} faces   | {(t1-t0)*1000:.1f} ms | {gpu_stats()}")

    # Save output
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), vis)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    if not client.is_server_live():
        print("ERROR: Triton server is not live!")
        return

    print("=" * 60)
    print("  Triton Inference Pipeline")
    print("  Person Detection → Face Detection → Face ReID")
    print("=" * 60)
    print(f"  Server  : {TRITON_URL}")
    print(f"  GPU     : {gpu_stats()}")
    print(f"  Input   : {INPUT_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 60)

    images = sorted(INPUT_DIR.glob("*.png")) + sorted(INPUT_DIR.glob("*.jpg"))
    if not images:
        print("No images found in input/")
        return

    for img_path in images:
        run_pipeline(client, img_path)

    print("\n" + "=" * 60)
    print(f"  Done — {len(images)} images processed")
    print(f"  Final GPU: {gpu_stats()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
