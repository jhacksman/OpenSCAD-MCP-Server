# Jetson Orin Nano and CUDA

The Jetson Orin Nano supports CUDA workloads beyond robotics. NVIDIA's [CUDA setup guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/setup_cuda.html) documents CUDA on the host and in Jetson-compatible containers. CUDA capability alone does not establish that an arbitrary project builds or runs correctly on the board.

## Current OpenSCAD server

The supported server does **not use CUDA**. It compiles SCAD and renders previews with OpenSCAD. The Docker image uses software rendering and does not require `--gpus`, an NVIDIA runtime, or JetPack CUDA packages.

The image has been built and tested on Linux arm64 in a VM on an Apple Silicon Mac. That verifies the ARM64 Linux container path, not physical Jetson hardware, JetPack integration, performance, or memory limits. Build it on the target using the [Docker instructions](docker.md) and run the container smoke test to validate that environment. Native installation requires Python 3.11+; if a board's OS has an older Python, use a compatible virtual environment or the container's bundled Python 3.13.

## Archived CUDA reconstruction prototype

The earlier README described remote CUDA reconstruction as supported. That was inaccurate: the integration generated artificial camera poses and wrote dummy OBJ geometry. It is now archived under `legacy/`, outside the installed server. Changing GPU hardware cannot fix that implementation.

The referenced [Fixstars CUMVS project](https://github.com/fixstars/cuda-multi-view-stereo#requirements) implements depth estimation and lists CUDA compute capability >= 6.0, OpenCV >= 4.6 with CUDA and Viz modules, and VTK >= 9.0. Its published benchmark uses an RTX 3080, not an Orin Nano. Those requirements and GPU benchmarks are not evidence of this repository's Jetson support.

Restoring a reconstruction feature would require a working ARM64 dependency build, real camera calibration/poses, actual mesh reconstruction, and tests with real image sets on the target GPU. No Jetson hardware or remote CUDA machine was used in this refresh, and no such support is claimed.
