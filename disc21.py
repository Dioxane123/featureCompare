import torch
import os
from PIL import Image
from timm.data.transforms_factory import create_transform
from timm.data import resolve_model_data_config

class DISC21Dataset(torch.utils.data.Dataset):
    """
    DISC21数据集加载类，用于加载指定目录下的所有图片并进行预处理。
    """
    def __init__(self, data_config, root_dir: str, query_path: str = ""):
        """
        Args:
        data_config: 模型数据配置，用于创建预处理变换
        root_dir(str): 图片所在目录路径
        query_path(str): 查询文件路径
        """
        self.root_path = root_dir
        self.transform = create_transform(**data_config, is_training=False)
        try:
            with open(query_path, 'r', encoding='utf-8') as f:
                self.image_files = [line.strip() + '.jpg' for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Query file {query_path} not found. Loading all images from {root_dir}.")
            self.image_files = [f for f in os.listdir(self.root_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.image_files[idx])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e
        if self.transform and not isinstance(self.transform, tuple):
            img = self.transform(img)
        return img, self.image_files[idx]