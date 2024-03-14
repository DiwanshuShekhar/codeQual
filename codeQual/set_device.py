import torch


def set() -> torch.device:
    device_string = "cpu"
    if torch.cuda.is_available():
        device_string = "cuda"
    elif torch.backends.mps.is_available():
        device_string = "mps"

    return torch.device(device_string)
