# C++ OpenCV Real-Time Vision Pipeline

A small **C++ + OpenCV** project demonstrating a complete classical computer-vision pipeline.

## Features

- Webcam or video-file input
- Grayscale conversion
- Gaussian filtering
- Otsu segmentation
- Contour detection
- Largest-object selection
- Rotated bounding-box geometry
- Object length/width measurement
- Real-time FPS display

## Why this project

The project is intentionally small but directly relevant to industrial computer vision. It shows that a complete vision solution can combine preprocessing, segmentation, geometry, decision logic, and performance measurement rather than relying only on a neural network.

## Requirements

- C++17
- CMake 3.16+
- OpenCV 4.x

## Linux setup

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Run with webcam

```bash
./vision_demo
```

## Run with video

```bash
./vision_demo /path/to/video.mp4
```

Press `q` or `Esc` to exit.

## Next improvements

Good next steps for making this more relevant to embedded/industrial vision:

- ROI selection
- Morphological filtering
- Optical flow
- Temporal filtering
- CSV export
- Runtime profiling
- Unit tests
- Multi-threading
- SIMD or GPU acceleration
