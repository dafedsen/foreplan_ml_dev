import torch

gpu_availability = torch.cuda.is_available()
print('GPU availability:', gpu_availability)

gpu_name = torch.cuda.get_device_name(0)
print('GPU name:', gpu_name)