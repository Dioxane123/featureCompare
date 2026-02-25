import numpy as np
import csv
from collections import defaultdict

ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'
LIST = '/media/dioxane/MovieDisk/Dataset/disc21/filter_final_gt.csv'
TASK = 'business_cards'
CLASS = 'Canon'
MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'
LAYER_IDX = 2

K = 5

# query_feat = np.load(f"result/{MODEL_NAME}/{TASK}/{CLASS}/layer_{LAYER_IDX}_features.npy")
query_feat = np.load(f"result/{MODEL_NAME}/{TASK}/{CLASS}/embeddings.npy")
query_label = np.load(f"result/{MODEL_NAME}/{TASK}/{CLASS}/files.npy", allow_pickle=True)
# gallery_feat = np.load(f"result/{MODEL_NAME}/{TASK}/References/layer_{LAYER_IDX}_features.npy")
gallery_feat = np.load(f"result/{MODEL_NAME}/{TASK}/Reference/embeddings.npy")
gallery_label = np.load(f"result/{MODEL_NAME}/{TASK}/Reference/files.npy", allow_pickle=True)
print(f"Query序列长度: {len(query_label)}, Gallery序列长度: {len(gallery_label)}")

gt_map = defaultdict(set)
try:
    with open(LIST, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                q_name = f"{row[0].strip()}.jpg"
                g_name = f"{row[1].strip()}.jpg"
                gt_map[q_name].add(g_name)
    print(f"已加载真值数据，共包含 {len(gt_map)} 个查询的真值信息。")
except Exception as e:
    print(f"读取真值文件失败: {e}")

similarity_matrix = np.dot(query_feat, gallery_feat.T)  # shape: [N_query, N_gallery]

def get_topk_similar_images(similarity_matrix, query_label, gallery_label, topk=5):
    topk_indices = np.argsort(-similarity_matrix, axis=1)[:, :topk]  # shape: [N_query, topk]
    results = {}
    for i, query in enumerate(query_label):
        results[query] = [(gallery_label[idx], similarity_matrix[i, idx]) for idx in topk_indices[i]]
    return results

results = get_topk_similar_images(similarity_matrix, query_label, gallery_label, topk=K)
accuracy_count = 0
# for query, sims in results.items():
#     correct_found = False
#     for gallery_img, score in sims:
#         if query == gallery_img:
#             correct_found = True
#             break
#     if correct_found:
#         accuracy_count += 1
# SMVS数据集：query文件名和gallery文件名相同就是匹配的真值
# 不需要额外的GT文件
for query, sims in results.items():
    correct_found = False
    for gallery_img, score in sims:
        if query == gallery_img:  # 文件名相同即为匹配
            correct_found = True
            break

    if correct_found:
        accuracy_count += 1


print(f"Top-{K}准确率: {accuracy_count}/{len(query_label)} = {accuracy_count / len(query_label):.4f}")