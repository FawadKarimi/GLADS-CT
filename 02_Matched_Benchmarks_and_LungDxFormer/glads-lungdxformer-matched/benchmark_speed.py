
import torch
import torchvision.models as models
import time

device = torch.device("cuda:1")
x = torch.randn(32, 1, 64, 64, device=device)

# Test ResNet-18
base_r = models.resnet18(weights=None)
base_r.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
base_r = base_r.to(device)

t0 = time.time()
with torch.cuda.amp.autocast():
    for _ in range(50):
        out = base_r(x)
torch.cuda.synchronize()
t_res = time.time() - t0

# Test ConvNeXt-Tiny
base_c = models.convnext_tiny(weights=None)
base_c.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=4, stride=4)
base_c = base_c.to(device)

t0 = time.time()
with torch.cuda.amp.autocast():
    for _ in range(50):
        out = base_c(x)
torch.cuda.synchronize()
t_conv = time.time() - t0

print(f"ResNet18 50 batches: {t_res:.4f}s | ConvNeXt 50 batches: {t_conv:.4f}s")

