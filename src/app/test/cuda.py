import torch

print(f"{torch.cuda.is_available()=}")
print(f"{torch.cuda.device_count()=}")
print(f"{torch.cuda.get_device_name(0)=}")
