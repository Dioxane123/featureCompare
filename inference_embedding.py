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

# 加载模型
# ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'
ROOT = '/media/dioxane/MovieDisk/Dataset/disc21/final_queries/images/final_queries'
LIST = '/media/dioxane/MovieDisk/Dataset/disc21/filter_queries.csv'
TASK = 'DISC21'
CLASS = 'Queries'
MODEL_NAME = 'vit_large_patch16_dinov3.lvd1689m'
BATCH_SIZE = 16
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device)
model.eval()

# 定义预处理
data_config = resolve_model_data_config(model)
transforms = create_transform(**data_config, is_training=False)

def get_image_embeddings(image_path, model):
    """
    读取图片，推理，并返回处理好的特征向量列表。
    """
    dataset = DISC21Dataset(data_config, image_path, LIST)
    # dataset = SMVSDataset(image_path, data_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"共发现{len(dataset)}张图片")
    
    file_paths_list = []
    embedding_buffer = []
    # 模型推理
    with torch.no_grad():
        for imgs, img_paths in tqdm(dataloader, desc="Processing Images"):
            imgs = imgs.to(device)
            embedding = model(imgs) # 得到[B, 768]的特征
            embedding = F.normalize(embedding, p=2, dim=1) # L2 Normalized
            file_paths_list.append(img_paths)

            embedding_buffer.append(embedding.cpu().numpy())# check the type here!
    return embedding_buffer, file_paths_list

# embedding_buffer, file_paths_list = get_image_embeddings(f"{ROOT}/{TASK}/{CLASS}", model)
embedding_buffer, file_paths_list = get_image_embeddings(ROOT, model)

os.makedirs(f"result/{MODEL_NAME}/{TASK}/{CLASS}", exist_ok=True)
file_array = np.concatenate(file_paths_list, axis=0) # shape: [N,]
embedding_array = np.concatenate(embedding_buffer, axis=0) # shape: [N, 768]
np.save(f"result/{MODEL_NAME}/{TASK}/{CLASS}/files.npy", file_array)
np.save(f"result/{MODEL_NAME}/{TASK}/{CLASS}/embeddings.npy", embedding_array)