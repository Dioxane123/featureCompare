import numpy as np
import os
from tqdm import tqdm


# 配置参数
MODEL_NAME = 'vit_large_patch14_clip_224.openai'

# 所有task列表
TASKS = ['business_cards', 'cd_covers', 'dvd_covers', 'landmarks', 'museum_paintings', 'print', 'video_frames']

# K值
K_VALUES = [1, 3]

# 结果目录
result_dir = f"result/{MODEL_NAME}"


def cosine_similarity(a, b):
    """
    计算余弦相似度
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def load_embeddings(task, subclass):
    """
    加载指定task和子类的embeddings和文件名
    """
    embeddings = np.load(os.path.join(result_dir, task, f"{subclass}_embeddings.npy"))
    files = np.load(os.path.join(result_dir, task, f"{subclass}_files.npy"), allow_pickle=True)
    return embeddings, files


def compute_top_k(query_embeddings, query_files, ref_embeddings, ref_files, k_values):
    """
    对每个query图片，查找top-k个最相似的reference图片
    """
    results = {k: [] for k in k_values}

    for i in range(len(query_embeddings)):
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


def evaluate_retrieval(results, query_subclass, k_values, ref_files):
    """
    评估检索结果
    假设query图片和对应reference图片有相同的编号
    """
    metrics = {}

    for k in k_values:
        correct = 0
        total = len(results[k])

        for result in results[k]:
            query_name = result['query']
            # 提取query的文件名（不含扩展名）
            query_base = os.path.splitext(query_name)[0]

            # 检查top-k中是否有匹配的文件名
            for ref_file in result['ref_files']:
                ref_base = os.path.splitext(ref_file)[0]
                if query_base == ref_base:
                    correct += 1
                    break

        metrics[f'top{k}'] = correct / total if total > 0 else 0
        metrics[f'top{k}_correct'] = correct
        metrics[f'top{k}_total'] = total

    # 计算mAP
    ap_sum = 0
    for result in results[max(k_values)]:
        query_name = result['query']
        query_base = os.path.splitext(query_name)[0]

        # 找到正确reference的索引
        correct_indices = []
        for j, ref_file in enumerate(ref_files):
            ref_base = os.path.splitext(ref_file)[0]
            if query_base == ref_base:
                correct_indices.append(j)

        if not correct_indices:
            continue

        # 计算每个正确结果在排序中的位置
        precisions = []
        for rank, (sim, ref_idx) in enumerate(zip(result['similarities'], result['top_k']), 1):
            if ref_idx in correct_indices:
                precisions.append(1.0 / rank)

        if precisions:
            ap_sum += max(precisions)

    metrics['mAP'] = ap_sum / len(results[max(k_values)]) if results[max(k_values)] else 0

    return metrics


def get_query_subclasses(task_dir):
    """
    获取该task的查询子类列表（排除Reference）
    从npy文件名中提取子类名
    """
    subclasses = []
    for item in os.listdir(task_dir):
        if item.endswith('_embeddings.npy'):
            subclass = item.replace('_embeddings.npy', '')
            if subclass != 'Reference':
                subclasses.append(subclass)
    return sorted(subclasses)


def process_task(task):
    """
    处理单个task的检索评估
    """
    task_dir = os.path.join(result_dir, task)
    if not os.path.exists(task_dir):
        print(f"  Warning: {task_dir} does not exist, skipping...")
        return None

    print(f"\nProcessing task: {task}")

    # 获取查询子类
    query_subclasses = get_query_subclasses(task_dir)
    print(f"  Query subclasses: {query_subclasses}")

    # 检查是否有Reference
    ref_path = os.path.join(task_dir, "Reference_embeddings.npy")
    if not os.path.exists(ref_path):
        print(f"  Warning: No Reference embeddings found, skipping...")
        return None

    # 加载reference embeddings
    ref_embeddings, ref_files = load_embeddings(task, 'Reference')
    print(f"  Reference: {len(ref_files)} images")

    # 汇总所有query的结果
    all_metrics = {
        'top1_correct': 0, 'top1_total': 0,
        'top3_correct': 0, 'top3_total': 0,
        'mAP_sum': 0, 'query_count': 0
    }

    for query_subclass in query_subclasses:
        query_path = os.path.join(task_dir, f"{query_subclass}_embeddings.npy")
        if not os.path.exists(query_path):
            continue

        # 加载query embeddings
        query_embeddings, query_files = load_embeddings(task, query_subclass)
        print(f"  Query {query_subclass}: {len(query_files)} images")

        # 计算top-k检索结果
        results = compute_top_k(query_embeddings, query_files, ref_embeddings, ref_files, K_VALUES)

        # 评估结果
        metrics = evaluate_retrieval(results, query_subclass, K_VALUES, ref_files)

        print(f"    Top-1: {metrics['top1_correct']}/{metrics['top1_total']} = {metrics['top1']:.4f}")
        print(f"    Top-3: {metrics['top3_correct']}/{metrics['top3_total']} = {metrics['top3']:.4f}")
        print(f"    mAP: {metrics['mAP']:.4f}")

        # 累加
        all_metrics['top1_correct'] += metrics['top1_correct']
        all_metrics['top1_total'] += metrics['top1_total']
        all_metrics['top3_correct'] += metrics['top3_correct']
        all_metrics['top3_total'] += metrics['top3_total']
        all_metrics['mAP_sum'] += metrics['mAP'] * len(query_files)
        all_metrics['query_count'] += len(query_files)

    # 计算总体指标
    overall = {
        'top1': all_metrics['top1_correct'] / all_metrics['top1_total'] if all_metrics['top1_total'] > 0 else 0,
        'top3': all_metrics['top3_correct'] / all_metrics['top3_total'] if all_metrics['top3_total'] > 0 else 0,
        'mAP': all_metrics['mAP_sum'] / all_metrics['query_count'] if all_metrics['query_count'] > 0 else 0
    }

    print(f"  ----------------------------------------")
    print(f"  Overall Top-1: {all_metrics['top1_correct']}/{all_metrics['top1_total']} = {overall['top1']:.4f}")
    print(f"  Overall Top-3: {all_metrics['top3_correct']}/{all_metrics['top3_total']} = {overall['top3']:.4f}")
    print(f"  Overall mAP: {overall['mAP']:.4f}")

    return overall


def main():
    print(f"{'='*70}")
    print(f"Retrieval Evaluation for {MODEL_NAME}")
    print(f"{'='*70}")

    task_results = {}

    for task in TASKS:
        result = process_task(task)
        if result:
            task_results[task] = result

    # 打印汇总表格
    print(f"\n{'='*70}")
    print(f"SUMMARY - {MODEL_NAME}")
    print(f"{'='*70}")
    print(f"{'Task':<20} {'Top-1':>10} {'Top-3':>10} {'mAP':>10}")
    print(f"{'-'*70}")

    total_top1 = 0
    total_top3 = 0
    total_mAP = 0
    count = 0

    for task in TASKS:
        if task in task_results:
            r = task_results[task]
            print(f"{task:<20} {r['top1']:>10.4f} {r['top3']:>10.4f} {r['mAP']:>10.4f}")
            total_top1 += r['top1']
            total_top3 += r['top3']
            total_mAP += r['mAP']
            count += 1

    print(f"{'-'*70}")
    if count > 0:
        print(f"{'Average':<20} {total_top1/count:>10.4f} {total_top3/count:>10.4f} {total_mAP/count:>10.4f}")

    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()