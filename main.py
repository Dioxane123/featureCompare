import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 1. 加载一张图片 (请替换为你的本地图片路径)
# 这里为了演示，假设有一张图片对象
# img = Image.open("path/to/your/image.jpg").convert('RGB')
# 如果没有图片，我们可以生成一个随机的伪造图片用于演示
img = Image.new('RGB', (224, 224), color='red')

# 2. 创建模型
# 'resnet50' 可以替换为其他模型名，如 'convnext_tiny', 'vit_base_patch16_224'
# pretrained=True: 加载在 ImageNet 上预训练的权重
# num_classes=0: 关键参数！这将移除全连接层，直接输出特征向量
model = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True, features_only=True)

# 3. 设置模型为推理模式
# 这会冻结 BatchNorm 层并禁用 Dropout，保证输出稳定
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# 5. 处理图片并增加 Batch 维度
# 输出形状通常为 [1, 3, H, W]
input_tensor = transforms(img).unsqueeze(0)

# 6. 提取特征
# 使用 no_grad 减少显存占用，因为我们不需要反向传播
with torch.no_grad():
    feature_vector = model(input_tensor)
    for feat in feature_vector:
        print(feat.shape)

# 7. 查看结果
# print(f"模型架构: {model.default_cfg['architecture']}")
# print(f"特征向量形状: {feature_vector.shape}") 
# 对于 ViT，通常输出是 [1, 768]