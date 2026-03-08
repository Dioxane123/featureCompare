import numpy as np
import os
from tqdm import tqdm


# 配置参数
DISC21_ROOT = '/media/dioxane/MovieDisk/Dataset/disc21'
GROUND_TRUTH_FILE = os.path.join(DISC21_ROOT, 'filter_final_gt.csv')

MODEL_NAME = 'vit_base_patch16_dinov3.lvd1689m'

# K值
K_VALUES = [1, 3]

# 结果目录
result_dir = f"result/{MODEL_NAME}/DISC21"


def cosine_similarity(a, b):
    """
    计算余弦相似度
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def load_ground_truth():
    """
    加载filter_final_gt.csv中的真值对应关系
    返回: dict {query_id: ref_id}
    """
    gt = {}
    with open(GROUND_TRUTH_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    query_id, ref_id = parts
                    gt[query_id] = ref_id
    return gt


def load_embeddings():
    """
    加载query和reference的embeddings和文件名
    """
    query_embeddings = np.load(os.path.join(result_dir, "query_embeddings.npy"))
    query_files = np.load(os.path.join(result_dir, "query_files.npy"), allow_pickle=True)

    ref_embeddings = np.load(os.path.join(result_dir, "reference_embeddings.npy"))
    ref_files = np.load(os.path.join(result_dir, "reference_files.npy"), allow_pickle=True)

    return query_embeddings, query_files, ref_embeddings, ref_files


def compute_top_k_retrieval(query_embeddings, query_files, ref_embeddings, ref_files, k_values):
    """
    对每个query图片，查找top-k个最相似的reference图片
    """
    results = {k: [] for k in k_values}

    # 构建ref_files的索引映射：文件名 -> 索引
    ref_file_to_idx = {ref_files[i]: i for i in range(len(ref_files))}

    for i in tqdm(range(len(query_embeddings)), desc="Computing retrieval"):
        query_emb = query_embeddings[i]
        query_name = query_files[i]

        # 计算与所有reference图片的相似度
        similarities = []
        for j in range(len(ref_embeddings)):
            sim = cosine_similarity(query_emb, ref_embeddings[j])
            similarities.append((sim, j))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 保存top-k结果
        for k in k_values:
            top_k_indices = [sim[1] for sim in similarities[:k]]
            results[k].append({
                'query': query_name,
                'top_k': top_k_indices,
                'ref_files': [ref_files[idx] for idx in top_k_indices],
                'similarities': [sim[0] for sim in similarities[:k]]
            })

    return results


def evaluate_retrieval(results, ground_truth, ref_files, k_values):
    """
    评估检索结果
    """
    metrics = {}

    # 构建ref_files的索引映射
    ref_file_to_idx = {ref_files[i]: i for i in range(len(ref_files))}

    for k in k_values:
        correct = 0
        total = len(results[k])

        for result in results[k]:
            query_name = result['query']
            # 获取该query的真实reference ID
            correct_ref = ground_truth.get(query_name)

            if correct_ref is None:
                continue

            # 检查top-k中是否包含正确的reference
            for ref_file in result['ref_files']:
                ref_base = os.path.splitext(ref_file)[0]
                # 比较时去掉扩展名，或者直接比较完整文件名
                if ref_base == correct_ref or ref_file == correct_ref + '.jpg':
                    correct += 1
                    break

        metrics[f'top{k}'] = correct / total if total > 0 else 0
        metrics[f'top{k}_correct'] = correct
        metrics[f'top{k}_total'] = total

    # 计算mAP
    ap_sum = 0
    valid_count = 0
    max_k = max(k_values)

    for result in results[max_k]:
        query_name = result['query']
        correct_ref = ground_truth.get(query_name)

        if correct_ref is None:
            continue

        valid_count += 1

        # 找到正确reference在ref_files中的索引
        correct_indices = set()
        for j, ref_file in enumerate(ref_files):
            ref_base = os.path.splitext(ref_file)[0]
            if ref_base == correct_ref or ref_file == correct_ref + '.jpg':
                correct_indices.add(j)

        if not correct_indices:
            continue

        # 计算每个正确结果在排序中的位置
        precisions = []
        for rank, ref_idx in enumerate(result['top_k'], 1):
            if ref_idx in correct_indices:
                precisions.append(1.0 / rank)

        if precisions:
            ap_sum += max(precisions)

    metrics['mAP'] = ap_sum / valid_count if valid_count > 0 else 0

    return metrics


def main():
    print(f"{'='*70}")
    print(f"DISC21 Retrieval Evaluation for {MODEL_NAME}")
    print(f"{'='*70}")

    # 加载ground truth
    print("\nLoading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Ground truth entries: {len(ground_truth)}")

    # 加载embeddings
    print("\nLoading embeddings...")
    query_embeddings, query_files, ref_embeddings, ref_files = load_embeddings()
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Reference embeddings shape: {ref_embeddings.shape}")

    # 计算top-k检索
    print("\nComputing retrieval...")
    results = compute_top_k_retrieval(query_embeddings, query_files, ref_embeddings, ref_files, K_VALUES)

    # 评估结果
    print("\nEvaluating results...")
    metrics = evaluate_retrieval(results, ground_truth, ref_files, K_VALUES)

    # 打印结果
    print(f"\n{'='*70}")
    print(f"RESULTS - {MODEL_NAME}")
    print(f"{'='*70}")
    print(f"Top-1: {metrics['top1_correct']}/{metrics['top1_total']} = {metrics['top1']:.4f}")
    print(f"Top-3: {metrics['top3_correct']}/{metrics['top3_total']} = {metrics['top3']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()