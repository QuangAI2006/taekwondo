import sys
import platform

print("=== System ===")
print("OS:", platform.platform())
print("Machine:", platform.machine())
print("Processor:", platform.processor())

try:
    import psutil
    mem_gb = psutil.virtual_memory().total / (1024**3)
    print("RAM: %.2f GB" % mem_gb)
except:
    print("RAM: unavailable")

print("\n=== Python ===")
print("Python:", sys.version)

try:
    import torch
    print("\n=== PyTorch ===")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch info unavailable:", e)

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except:
    pass

try:
    import mediapipe as mp
    print("MediaPipe:", mp.__version__)
except:
    pass