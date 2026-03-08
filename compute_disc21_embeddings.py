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
DISC21_ROOT = '/media/dioxane/MovieDisk/Dataset/disc21'
QUERY_DIR = os.path.join(DISC21_ROOT, 'final_queries/images/final_queries')
REF_DIR = os.path.join(DISC21_ROOT, 'references_0/images/references')
FILTER_QUERIES = os.path.join(DISC21_ROOT, 'filter_queries.csv')

MODEL_NAME = 'vit_medium_patch16_clip_224.tinyclip_yfcc15m'
BATCH_SIZE = 8  # 减小batch size以避免内存问题

# 加载模型
print(f"Loading model: {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device)
model.eval()

# 定义预处理
data_config = resolve_model_data_config(model)
transforms = create_transform(**data_config, is_training=False)


class ImageDataset(torch.utils.data.Dataset):
    """
    流式加载图片的Dataset，避免一次性加载所有图片到内存
    """
    def __init__(self, image_dir, filter_set=None, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.transform = transforms

        # 获取所有符合条件的图片文件
        all_files = [f for f in os.listdir(image_dir)
                     if f.lower().endswith(extensions) and f != 'Thumbs.db']

        # 如果有filter_set，则过滤
        if filter_set is not None:
            self.image_files = []
            for f in all_files:
                file_id = os.path.splitext(f)[0]
                if file_id in filter_set:
                    self.image_files.append(f)
        else:
            self.image_files = all_files

        # 排序以保证一致性
        self.image_files.sort()
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            file_id = os.path.splitext(img_file)[0]
            return img_tensor, file_id
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回空图片
            img_tensor = torch.zeros(3, 224, 224)
            return img_tensor, os.path.splitext(img_file)[0]


def get_filtered_queries():
    """
    从filter_queries.csv读取需要处理的query ID列表
    """
    with open(FILTER_QUERIES, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    return set(queries)


def process_images_streaming(image_dir, filter_set=None, batch_size=8):
    """
    流式处理图片，边加载边推理，不一次性加载所有图片到内存
    """
    dataset = ImageDataset(image_dir, filter_set)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    embeddings_list = []
    file_names_list = []

    with torch.no_grad():
        for imgs, names in tqdm(dataloader, desc="Processing"):
            imgs = imgs.to(device)
            embeddings = model(imgs)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            embeddings_list.append(embeddings.cpu().numpy())
            file_names_list.extend(names)

    embeddings = np.concatenate(embeddings_list, axis=0)
    return embeddings, file_names_list


def main():
    # 创建结果目录
    result_dir = f"result/{MODEL_NAME}/DISC21"
    os.makedirs(result_dir, exist_ok=True)

    # 加载需要处理的query ID列表
    filtered_queries = get_filtered_queries()
    print(f"Filtered queries count: {len(filtered_queries)}")

    # 1. 处理Reference图片（流式处理）
    print(f"\n{'='*60}")
    print("Processing Reference images...")
    print(f"{'='*60}")
    ref_embeddings, ref_files = process_images_streaming(REF_DIR, filter_set=None, batch_size=BATCH_SIZE)
    print(f"Reference embeddings shape: {ref_embeddings.shape}")

    np.save(os.path.join(result_dir, "reference_embeddings.npy"), ref_embeddings)
    np.save(os.path.join(result_dir, "reference_files.npy"), np.array(ref_files))

    # 2. 处理Query图片（仅处理filter_queries.csv中的）
    print(f"\n{'='*60}")
    print("Processing Query images...")
    print(f"{'='*60}")
    query_embeddings, query_files = process_images_streaming(QUERY_DIR, filter_set=filtered_queries, batch_size=BATCH_SIZE)
    print(f"Query embeddings shape: {query_embeddings.shape}")

    np.save(os.path.join(result_dir, "query_embeddings.npy"), query_embeddings)
    np.save(os.path.join(result_dir, "query_files.npy"), np.array(query_files))

    print(f"\n{'='*60}")
    print("All embeddings computed and saved!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()