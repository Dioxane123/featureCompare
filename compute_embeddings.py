import numpy as np
import os
import torch
import timm
from tqdm import tqdm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import torch.nn.functional as F
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置参数
ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'
TASK = 'business_cards'
MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'
BATCH_SIZE = 16

# 子类列表（包含Reference）
SUBCLASES = ['Canon', 'Droid', 'E63', 'Palm', 'Reference']

# 加载模型
print(f"Loading model: {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device)
model.eval()

# 定义预处理
data_config = resolve_model_data_config(model)
transforms = create_transform(**data_config, is_training=False)


def load_and_process_images(image_dir, transforms):
    """
    读取目录下所有图片并处理成tensor列表
    """
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and f != 'Thumbs.db']
    image_files.sort()

    images = []
    file_names = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transforms(img)
            images.append(img_tensor)
            file_names.append(img_file)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return images, file_names


def get_embeddings(images, model, batch_size=32):
    """
    批量获取图片的embeddings
    """
    embeddings = []
    embedding_buffer = []

    # 创建 DataLoader
    dataset = torch.utils.data.TensorDataset(torch.stack(images))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            imgs = batch[0].to(device)
            embedding = model(imgs)
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding_buffer.append(embedding.cpu().numpy())

    embedding_array = np.concatenate(embedding_buffer, axis=0)
    return embedding_array


# 创建结果目录
result_dir = f"result/{MODEL_NAME}/{TASK}"
os.makedirs(result_dir, exist_ok=True)

# 遍历每个子类，计算embeddings
for subclass in SUBCLASES:
    subclass_dir = os.path.join(ROOT, TASK, subclass)
    if not os.path.exists(subclass_dir):
        print(f"Warning: {subclass_dir} does not exist, skipping...")
        continue

    print(f"\nProcessing {subclass}...")
    images, file_names = load_and_process_images(subclass_dir, transforms)
    print(f"  Found {len(images)} images")

    if len(images) == 0:
        continue

    embeddings = get_embeddings(images, model, BATCH_SIZE)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 保存embeddings和文件名
    np.save(os.path.join(result_dir, f"{subclass}_embeddings.npy"), embeddings)
    np.save(os.path.join(result_dir, f"{subclass}_files.npy"), np.array(file_names))

print("\nAll embeddings computed and saved!")