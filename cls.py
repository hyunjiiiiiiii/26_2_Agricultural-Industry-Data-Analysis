import torch
ckpt = torch.load("runs_cls/runs_cls/stn_resnet18/best.pth", map_location="cpu")
print("best epoch:", ckpt["epoch"])
print("classes:", ckpt["classes"])
