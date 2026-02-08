import numpy as np
import os
import torch
import timm
from tqdm import tqdm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import torch.nn.functional as F
from smvs import SMVSDataset
from disc21 import DISC21Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型
# ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'
ROOT = '/media/dioxane/MovieDisk/Dataset/disc21/references_0/images/references'
# LIST = '/media/dioxane/MovieDisk/Dataset/disc21/filter_final_queries.csv'
# TASK = 'business_cards'
TASK = 'DISC21'
CLASS = 'References'
MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'
BATCH_SIZE = 16
model = timm.create_model(MODEL_NAME, pretrained=True, features_only=True).to(device)
model.eval()

# 定义预处理
data_config = resolve_model_data_config(model)
transforms = create_transform(**data_config, is_training=False)

def get_image_embeddings(image_path, model):
    """
    读取图片，推理，并返回处理好的特征向量列表。
    """
    # dataset = SMVSDataset(image_path, data_config)
    dataset = DISC21Dataset(data_config, image_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"共发现{len(dataset)}张图片")
    
    features_buffer = {} #key: layer_idx, value: list of features
    file_paths_list = []
    # 模型推理
    with torch.no_grad():
        for imgs, img_paths in tqdm(dataloader, desc="Processing Images"):
            imgs = imgs.to(device)
            features_list = model(imgs) # 得到三个[B, 768, 16, 16]的特征列表
            file_paths_list.append(img_paths)
            for i, feat_map in enumerate(features_list):
                if i not in features_buffer:
                    features_buffer[i] = []
                # feat_map shape: [B, 768, 16, 16]
                pooled_feat = torch.mean(feat_map, dim=(2, 3)) # 全局平均池化->[B, 768]
                normalized_feat = F.normalize(pooled_feat, p=2, dim=1) # L2 Normalized
                features_buffer[i].append(normalized_feat.cpu().numpy())# check the type here!
    return features_buffer, file_paths_list

# features_buffer, file_paths_list = get_image_embeddings(f"{ROOT}/{TASK}/{CLASS}", model)
features_buffer, file_paths_list = get_image_embeddings(ROOT, model)

os.makedirs(f"result/{MODEL_NAME}/{TASK}/{CLASS}", exist_ok=True)
file_array = np.concatenate(file_paths_list, axis=0) # shape: [N,]
np.save(f"result/{MODEL_NAME}/{TASK}/{CLASS}/files.npy", np.array(file_array, dtype=object))
for layer_idx, feats in features_buffer.items():
    feats_array = np.concatenate(feats, axis=0) # shape: [N, 768]
    np.save(f"result/{MODEL_NAME}/{TASK}/{CLASS}/layer_{layer_idx}_features.npy", feats_array)