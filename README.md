# evalita2026-multipride-UniBO-FICLIT

## Introduction

This repository contains the code and resources for the UniBO-FICLIT system
submitted to the MultiPRIDE task at EVALITA 2026. The system focuses on
automatic detection of reclamation of slurs in LGBTQ+ contexts.

## Installation

This project uses `uv` (a fast Python package and project manager). Follow the
official installation instructions here:
https://docs.astral.sh/uv/getting-started/installation/.

After installing `uv`, choose whether to install PyTorch for CPU-only execution
or with CUDA 11.7 support. Experiments were run locally with an NVIDIA RTX 3060
and CUDA 11.7; if your GPU is compatible with CUDA 11.7, from the repository
root run:

```
uv sync --extra cu117
```

If you prefer CPU-only execution, run:

```
uv sync --extra cpu
```

Note: CUDA binaries require an NVIDIA GPU. macOS and AMD GPUs are not supported
by the CUDA option.

After installing the dependencies you should be able to open and run the
notebooks in the `notebooks/` directory.

## License

The training data are available under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
and were created by the organizers of the MultiPRIDE task at EVALITA 2026.
