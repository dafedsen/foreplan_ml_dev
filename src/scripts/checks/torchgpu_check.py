import torch

print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU name:", torch.cuda.get_device_name(0))

x = torch.rand(10000, 10000, device="cuda")
y = torch.mm(x, x)
print("Matrix multiply done on:", y.device)