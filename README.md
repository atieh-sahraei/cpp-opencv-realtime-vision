# C++ OpenCV Real-Time Vision Pipeline

A computer-vision pipeline for real-time video analysis.

The project combines: preprocessing, segmentation, geometric measurement, motion estimation, temporal filtering, and runtime profiling.

## Features

- Webcam or video-file input
- Grayscale conversion and Gaussian filtering
- Otsu thresholding
- Contour detection and largest-object selection
- Rotated bounding-box geometry using `minAreaRect`
- Object length and width measurement
- Dense optical flow using the Farnebäck method
- Sparse motion-vector visualization
- Mean optical-flow magnitude as a simple motion signal
- Exponential moving-average temporal filtering
- Per-stage runtime profiling
- Processing-FPS measurement

## Pipeline

```text
Camera / video
      |
      v
Grayscale + Gaussian blur
      |
      +----------------------+
      |                      |
      v                      v
Otsu segmentation        Dense optical flow
      |                      |
      v                      v
Contours + geometry      Motion magnitude
      |                      |
      |                      v
      |                Temporal smoothing
      |                      |
      +----------+-----------+
                 |
                 v
        Overlay + profiling
```

## Optical flow

The application uses OpenCV's dense Farnebäck optical flow:

```cpp
cv::calcOpticalFlowFarneback(...)
```

For each pair of consecutive grayscale frames, the algorithm estimates a 2D motion vector for every pixel.

The horizontal and vertical components are converted to magnitude and direction. The current demo uses the **mean magnitude over the frame** as a compact global motion measurement.

Sparse arrows are drawn on the displayed image so that the estimated motion field can be visually inspected.

### Important limitation

The current mean-motion value is calculated over the complete frame. This makes the example easy to understand, but background motion can influence the result. A more application-specific system would normally calculate motion inside an ROI or segmentation mask.

## Temporal filtering

Raw frame-to-frame measurements can fluctuate, so the motion signal is stabilized with an exponential moving average:

```text
smoothed = alpha * current + (1 - alpha) * previous
```

The current value is:

```cpp
alpha = 0.15
```

A smaller value gives stronger smoothing but a slower response. A larger value reacts faster but retains more frame-to-frame variation.

Both the raw and smoothed motion values are displayed so their behavior can be compared directly.

## Runtime profiling

The application measures the execution time of the main stages using `std::chrono::steady_clock`:

- preprocessing
- segmentation and geometry
- optical flow
- total frame processing

The overlay shows the timing in milliseconds and calculates the approximate processing throughput:

```text
Processing FPS = 1000 / total processing time in ms
```

This makes it possible to see which stage is the main computational bottleneck before attempting optimization.

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
git clone https://github.com/atieh-sahraei/cpp-opencv-realtime-vision.git
cd cpp-opencv-realtime-vision
mkdir build
cd build
cmake ..
cmake --build .
```

## Run with a webcam

```bash
./vision_demo
```

## Run with a video

```bash
./vision_demo /path/to/video.mp4
```

Press `q` or `Esc` to stop.

## What the display shows

The main visualization contains:

- detected object's rotated bounding box
- measured length and width
- optical-flow vectors
- raw mean motion in pixels/frame
- temporally smoothed motion in pixels/frame
- processing FPS
- preprocessing, segmentation, optical-flow, and total processing times

A second window displays the binary segmentation result.

## Project structure

```text
cpp-opencv-realtime-vision/
├── CMakeLists.txt
├── README.md
├── .gitignore
└── src/
    └── main.cpp
```

## Engineering choices

The objective is to make each stage easy to inspect and modify before moving toward more optimized embedded implementations.

The runtime measurements also provide a baseline for later optimization work. For this pipeline, dense optical flow is expected to be considerably more computationally expensive than simple thresholding and contour analysis, which makes it a natural target for optimization experiments.

## Possible next steps

- Restrict optical flow to an ROI or detected-object mask
- Add morphological filtering to improve segmentation
- Export measurements and timing data to CSV
- Compare dense and sparse optical-flow approaches
- Add unit tests for geometry and filtering logic
- Separate the pipeline into reusable C++ classes
- Add multi-threaded capture/processing
- Benchmark different image resolutions
- Investigate OpenCV optimization options, SIMD, or GPU acceleration
- Test on an embedded Linux target

