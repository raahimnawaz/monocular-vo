# Jetson Orin Nano Super deployment plan

Target: real-time monocular visual-inertial odometry on a Jetson Orin Nano Super (67 TOPS), using:
- This repo's VO + SLAM pipeline (C++-ported, TensorRT-accelerated depth)
- The C++ EKF from [vehicle-dynamics-estimation](https://github.com/raahimnawaz/vehicle-dynamics-estimation) as the IMU fusion layer
- A USB webcam + MPU6050 IMU breakout

End state: ~25-30 fps live VIO on the board, with a 30-second demo video showing the pipeline running on real hardware.

---

## Hardware checklist (before the board lands)

- Jetson Orin Nano Super dev kit (~$249)
- Active cooling fan (the older Nano carrier throttles without one, ~$15)
- 64+ GB microSD or M.2 NVMe (~$20-30)
- USB webcam, UVC-compliant — Logitech C920 / Razer Kiyo / any modern USB cam (~$30)
- **MPU6050 IMU breakout** (~$10) — *not* BNO055; we want raw IMU, not on-board fusion (the EKF does the fusion)
- USB-C power supply (5 V / 5 A; quality matters — cheap supplies cause boot loops)
- Optional: a small 3D-printed mount to keep the camera + IMU rigidly fixed relative to each other for extrinsic calibration stability

## Software setup (Day 0 — first evening)

1. **Flash JetPack 6.2+** (TensorRT 10.x for Orin Super) via NVIDIA SDK Manager (needs an Ubuntu host) or community balena-etcher images
2. **Verify the bundled CUDA/cuDNN/TensorRT samples** run cleanly (~30 min sanity check)
3. **Install Jetson-specific PyTorch wheels** from `developer.nvidia.com/embedded/jetson-pytorch` — the generic `pip install torch` won't see CUDA
4. **Smoke-test PyTorch CUDA** with one tensor multiply and one `torch.cuda.is_available()` confirm
5. **Clone monocular-vo + vehicle-dynamics-estimation** onto the board

## Phase 1 — verify Python pipeline runs on CUDA (~1 evening)

- Run `scripts/run_vo.py` on a transferred test video using the existing Python pipeline, with `--depth-model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` on CUDA
- Expected: ~50-80 ms depth inference per frame (PyTorch baseline; pre-TensorRT)
- Confirms environment is sound before any C++ work begins

## Phase 2 — Depth Anything v2 → ONNX → TensorRT FP16 (~1-2 days)

Goal: drop depth inference from ~60 ms (PyTorch CUDA) to **15-25 ms** (TensorRT FP16).

Steps:
1. Add `scripts/export_depth_to_onnx.py` — uses HuggingFace `transformers` to load the model, then `torch.onnx.export` with static input shape (e.g. 644×476 → divisible by 14 for DPT)
2. Verify with `onnxruntime` that the ONNX model produces the same depth as the PyTorch model (atol 1e-3)
3. Build TensorRT engine: `trtexec --onnx=depth_anything_v2_small.onnx --fp16 --saveEngine=depth.trt --workspace=2048`
4. Add a C++ wrapper that loads the .trt engine, runs inference via the TensorRT runtime API, and writes the output depth map
5. Benchmark FP32 → FP16 → (optionally) INT8 with calibration on a few hundred frames; pick the fastest variant that keeps scale accuracy reasonable

Risks:
- Op compatibility (some transformer ops don't ONNX-export cleanly; may need `opset_version=17` and to fall back to `dynamo_export`)
- Dynamic shapes — fix at one resolution to simplify
- INT8 calibration drift — TUM and the user's own captures are the calibration set

## Phase 3 — C++ port of the VO front-end (~1 weekend)

Translate `src/monocular_vo/vo.py` to C++17:
- ORB detect + describe via OpenCV C++ (`cv::ORB::create`, `cv::BFMatcher`)
- Lowe's ratio test
- Depth backprojection (3D point cloud from matched keypoints + depth)
- `cv::solvePnPRansac` for relative pose
- Trajectory accumulator (same SE(3) composition as the Python `Trajectory` class)

Layout under `cpp/`:
```
cpp/
├── include/monocular_vo/
│   ├── vo.hpp
│   ├── depth_trt.hpp        # TensorRT depth wrapper
│   └── trajectory.hpp
├── src/                     # .cpp implementations
├── tests/                   # Catch2 or doctest unit tests
└── CMakeLists.txt
```

Targets:
- VO step (features + match + PnP): < 8 ms
- Header-only where reasonable, allocation-free in the hot loop (mirror the vehicle-dynamics style)

## Phase 4 — gtsam C++ pose graph (~1-2 days)

gtsam is already C++ — the Python `slam.py` is just bindings over the same library. Port:
- `PoseGraph` class using `gtsam::NonlinearFactorGraph`, `gtsam::Values`, `gtsam::LevenbergMarquardtOptimizer`
- Same `BetweenFactorPose3` for odometry + loop closures
- Same `pnp_to_relative_pose` conversion (documented inline in the Python version — math is identical)

Verify by feeding the same synthetic 4-pose square-loop test data and checking the optimized output matches Python to 1e-9 m.

## Phase 5 — Python ↔ C++ parity tests (~1 day)

Goal: same input video, same calibration, identical trajectory output between the Python and C++ pipelines, to 1e-9 m at every step.

Approach:
- Drive both pipelines from the same recorded depth maps (avoid re-running the depth model so the comparison isolates the VO/SLAM code)
- A small `pytest` + `subprocess` harness runs the C++ binary on the same input and compares the position CSVs

Failure modes to expect:
- Floating-point sum-order differences from RANSAC's randomness (seed it)
- BFMatcher ties broken differently in OpenCV C++ vs Python (sort matches by distance + index for determinism)

## Phase 6 — submodule vde, wire VIO (~1 weekend; extrinsic calibration is the time sink)

- `git submodule add` the vehicle-dynamics-estimation repo under `cpp/third_party/vehicle_dynamics_ekf/`
- Use its allocation-free EKF as the IMU integration layer
- Modify the state vector to include camera pose + IMU bias terms
- VO emits keyframe-rate pose corrections (15-30 Hz); IMU runs free between keyframes (200 Hz)
- Time-sync IMU and camera (timestamps must be aligned; use the host clock or a hardware sync if possible)

**Extrinsic calibration** (camera ↔ IMU 6-DoF transform) is the painful part:
- Use [Kalibr](https://github.com/ethz-asl/kalibr) — academic standard, works but rough edges
- Alternative: estimate online as part of the state vector — more work to implement but no offline calibration step needed
- Plan: 1-2 days of fighting Kalibr, then it works for the demo

## Phase 7 — live on Jetson with USB webcam + IMU (~1 day)

- Wire MPU6050 over I2C to Jetson 40-pin header (SDA/SCL pins)
- Write a small IMU reader that streams to a memory queue at 200 Hz
- Hook USB webcam via OpenCV `VideoCapture`
- Run the C++ binary live, render a top-down trajectory in real time (or save to disk and replay)
- Record a 30-second demo video showing: walk around a desk → return to start → loop closure correction visible on the live plot

## Phase 8 — KITTI seq 00 (~1 day on Jetson, separate from VIO demo)

Run the Python `scripts/run_kitti.py` on KITTI Odometry sequence 00 using the **outdoor** Depth Anything variant (`depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`). The TUM benchmark covered the indoor case; KITTI covers outdoor street footage.

Steps:
1. Download KITTI Odometry color images (~5 GB for seq 00 alone, or 65 GB for all). Register at cvlibs.net first (email-gated, free).
2. Lay out as expected: `<root>/sequences/00/image_2/*.png`, `<root>/sequences/00/calib.txt`, `<root>/poses/00.txt`
3. Run: `uv run python scripts/run_kitti.py --root <kitti_root> --seq 00 --max-frames 1000` (start with 1000 frames to verify; KITTI seq 00 is 4541 frames total)
4. Once verified, run full sequence — expect ~15-20 min on Orin GPU with TensorRT depth, longer on PyTorch
5. Report ATE/RPE in the README alongside the TUM numbers
6. Compare against published baselines for context (ORB-SLAM2 ATE on KITTI 00 ≈ 1.3 m; DSO ≈ 0.9 m; DeepVO ≈ 32 m; ours will likely land between these depending on depth model accuracy)

Outdoor-specific risks:
- Depth Anything's outdoor variant may give larger scale errors on very long-range scenes (>50 m)
- Sky and dynamic objects (cars, pedestrians) violate the static-scene assumption — RANSAC handles some but not extreme cases
- 4541 frames at 15 ms depth + 8 ms VO = ~104 s on Jetson with TensorRT; ~20 min on M5 with PyTorch

## Phase 9 — documentation + benchmarks page (~1 day)

This is what recruiters actually click on. Write up:
- `docs/benchmarks.md` — per-stage latency table, accuracy on TUM + KITTI, comparison to baselines
- Embed the Jetson demo video (uploaded as GitHub release asset for inline playback)
- Update the main README headline: "**Real-time monocular VIO on a $249 Jetson Orin Nano Super.** Foundation-model depth (TensorRT FP16, 15-25 ms) + classical VO + gtsam pose graph + my own C++ EKF for IMU fusion."
- Cross-link from vehicle-dynamics-estimation's README ("the EKF here powers the IMU fusion in monocular-vo's Jetson deployment")
- Profile README update to mention the Jetson demo

---

## Total budget

| Phase | Time | Cost |
|---|---|---|
| 1. PyTorch on CUDA verification | 1 evening | — |
| 2. ONNX + TensorRT FP16 | 1-2 days | — |
| 3. C++ VO front-end | 1 weekend | — |
| 4. gtsam C++ pose graph | 1-2 days | — |
| 5. Py↔C++ parity tests at 1e-9 m | 1 day | — |
| 6. Submodule vde EKF + wire VIO | 1 weekend | — |
| 7. Live Jetson demo with webcam + IMU | 1 day | — |
| 8. KITTI seq 00 evaluation | 1 day | ~5 GB download |
| 9. Documentation + benchmarks page | 1 day | — |
| **Total** | **~140-225 hours / 3-6 weeks FTE** | Jetson + accessories ≈ $300 |

## End state ("A+" portfolio grade)

A single sentence the README can lead with:
> *Built monocular visual-inertial SLAM from scratch — foundation-model depth (TensorRT FP16, 15-25 ms on Orin) + classical multi-view geometry + my own gtsam-based pose graph with loop closure + my own allocation-free C++ EKF for IMU fusion. Benchmarked at 36 % ATE reduction on TUM RGB-D fr1_room and competitive numbers on KITTI seq 00. Runs at 25-30 fps live on a $249 Jetson Orin Nano Super.*

That sentence does serious work in an interview. It covers perception, classical robotics, embedded ML, and real hardware in one breath.
