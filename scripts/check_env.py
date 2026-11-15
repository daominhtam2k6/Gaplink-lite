import os
import torch

def main():
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU:", p.name, "| VRAM GB:", round(p.total_memory / 1e9, 2))
        x = torch.randn(2000, 2000, device="cuda")
        y = x @ x
        print("matmul ok:", tuple(y.shape))
    print("CPU cores:", os.cpu_count())

if __name__ == "__main__":
    main()
