import torch


def get_device() -> torch.device:
    """
    Detects and returns the appropriate device (GPU or CPU).

    Automatically checks if CUDA is available and returns a GPU device,
    otherwise falls back to CPU.

    Returns:
        torch.device: The device to use for model training/inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        return device
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available. Using CPU (training may be slow).")
        return device


def get_device_info() -> dict:
    """
    Returns comprehensive device information for logging and debugging.

    Returns:
        dict: Dictionary containing device info (device type, GPU info if available, etc.)
    """
    info = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()

    return info


def setup_device() -> torch.device:
    """
    Initializes and configures the device for training/inference.

    Sets up CUDA-specific optimizations if GPU is available:
    - Enables cuDNN auto-tuner for performance
    - Sets memory growth to avoid OOM issues

    Returns:
        torch.device: The configured device
    """
    device = get_device()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        print("✅ CUDA optimizations enabled")

    return device
