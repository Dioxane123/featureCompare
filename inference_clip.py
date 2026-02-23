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

# ==================== 配置参数 ====================
# 选择数据集: 'SMVS' 或 'DISC21'
DATASET = 'business_cards'

# 选择任务类型: 'Queries' 或 'References'
TASK_TYPE = 'Reference'


# 选择CLIP模型 (从clips.txt中选择)
MODEL_NAME = 'vit_base_patch16_clip_224.openai'

# 批处理大小
BATCH_SIZE = 16

# ==================== 数据路径 ====================
# SMVS数据集路径
SMVS_ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'

# DISC21数据集路径
DISC21_REFERENCES_ROOT = '/media/dioxane/MovieDisk/Dataset/disc21/references_0/images/references'
DISC21_QUERIES_ROOT = '/media/dioxane/MovieDisk/Dataset/disc21/final_queries/images/final_queries'
DISC21_QUERY_LIST = '/media/dioxane/MovieDisk/Dataset/disc21/filter_queries.csv'


def load_model(model_name: str):
    """
    加载CLIP模型

    Args:
        model_name: 模型名称

    Returns:
        model: 加载好的模型
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    return model


def get_preprocessing(model):
    """
    获取模型的数据预处理配置

    Args:
        model: 加载好的模型

    Returns:
        data_config: 数据配置
        transforms: 预处理变换
    """
    data_config = resolve_model_data_config(model)
    transforms = create_transform(**data_config, is_training=False)
    return data_config, transforms


def get_image_embeddings(image_path: str, model, data_config, dataset_class, query_list: str = ""):
    """
    读取图片，推理，并返回处理好的特征向量列表。

    Args:
        image_path: 图片目录路径
        model: 加载好的模型
        data_config: 数据配置
        dataset_class: 数据集类 (SMVSDataset 或 DISC21Dataset)
        query_list: 查询列表文件路径 (仅DISC21需要)

    Returns:
        embedding_buffer: 特征向量列表
        file_paths_list: 文件路径列表
    """
    if dataset_class == 'DISC21':
        dataset = DISC21Dataset(data_config, image_path, query_list)
    else:  # SMVS
        dataset = SMVSDataset(image_path, data_config)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"共发现 {len(dataset)} 张图片")

    file_paths_list = []
    embedding_buffer = []

    # 模型推理
    with torch.no_grad():
        for imgs, img_paths in tqdm(dataloader, desc="Processing Images"):
            imgs = imgs.to(device)
            embedding = model(imgs)  # 得到特征向量
            embedding = F.normalize(embedding, p=2, dim=1)  # L2 Normalized
            file_paths_list.append(img_paths)
            embedding_buffer.append(embedding.cpu().numpy())

    return embedding_buffer, file_paths_list


def save_results(embeddings, file_paths, model_name, dataset, task_type):
    """
    保存结果到文件

    Args:
        embeddings: 特征向量
        file_paths: 文件路径列表
        model_name: 模型名称
        dataset: 数据集名称
        task_type: 任务类型
    """
    output_dir = f"result/{model_name}/{dataset}/{task_type}"
    os.makedirs(output_dir, exist_ok=True)

    file_array = np.concatenate(file_paths, axis=0)  # shape: [N,]
    embedding_array = np.concatenate(embeddings, axis=0)  # shape: [N, feature_dim]

    np.save(f"{output_dir}/files.npy", file_array)
    np.save(f"{output_dir}/embeddings.npy", embedding_array)
    print(f"结果已保存到 {output_dir}")


def main():
    # 加载模型
    print(f"Loading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME)

    # 获取预处理配置
    data_config, transforms = get_preprocessing(model)

    # 根据配置选择数据路径
    if DATASET == 'DISC21':
        if TASK_TYPE == 'References':
            image_path = DISC21_REFERENCES_ROOT
        else:
            image_path = DISC21_QUERIES_ROOT
        query_list = DISC21_QUERY_LIST if TASK_TYPE == 'Queries' else ""
    else:  # SMVS
        image_path = f"{SMVS_ROOT}/{DATASET}/{TASK_TYPE}"
        query_list = ""

    print(f"Processing {DATASET} dataset, {TASK_TYPE}")
    print(f"Image path: {image_path}")

    # 提取特征
    embeddings, file_paths = get_image_embeddings(
        image_path=image_path,
        model=model,
        data_config=data_config,
        dataset_class=DATASET,
        query_list=query_list
    )

    # 保存结果
    save_results(embeddings, file_paths, MODEL_NAME, DATASET, TASK_TYPE)


if __name__ == '__main__':
    main()
