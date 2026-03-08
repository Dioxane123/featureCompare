import timm

# 加载模型 (如果只需要算参数量，可以设置 pretrained=False 以加快加载速度)
model = timm.create_model('vit_huge_plus_patch16_dinov3.lvd1689m', pretrained=False)

# 1. 计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量 (Total Parameters): {total_params:,}")

# 2. 计算可训练参数量 (微调时非常有用)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量 (Trainable Parameters): {trainable_params:,}")