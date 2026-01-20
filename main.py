import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True, features_only=True)
model.to(device)
model.eval()

img_path = '/media/dioxane/MovieDisk/Dataset/SMVS/business_cards/Canon/001.jpg'
img = Image.open(img_path).convert('RGB')


data_config = resolve_model_data_config(model)
transforms = create_transform(**data_config, is_training=False)

input_tensor = transforms(img).unsqueeze(0).to(device)

with torch.no_grad():
    feature_vector = model(input_tensor)
# for feat in feature_vector:
#     print(feat.shape)
for idx, feature in enumerate(feature_vector):
    feature = feature.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() # [16, 16, 768]
    pca_features = pca.fit_transform(feature.reshape(-1, feature.shape[-1])).reshape(16, 16, 3)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

    plt.figure(figsize=(5, 5))
    plt.imshow(pca_features)
    plt.axis('off')
    plt.title(f'Feature Map - Layer -{idx+1}') # 倒数第几层
    save_name = f'feature_vis_layer_{idx+1}.png'
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"已保存可视化图片: {save_name}")
    