import numpy as np
import os
from tqdm import tqdm


# 配置参数
ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'
TASK = 'business_cards'
MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'

# 子类列表（Query使用除Reference外的子类）
QUERY_SUBCLASES = ['Canon', 'Droid', 'E63', 'Palm']
REF_SUBCLASS = 'Reference'

# K值
K_VALUES = [1, 3]

# 结果目录
result_dir = f"result/{MODEL_NAME}/{TASK}"


def cosine_similarity(a, b):
    """
    计算余弦相似度
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def load_embeddings(subclass):
    """
    加载指定子类的embeddings和文件名
    """
    embeddings = np.load(os.path.join(result_dir, f"{subclass}_embeddings.npy"))
    files = np.load(os.path.join(result_dir, f"{subclass}_files.npy"), allow_pickle=True)
    return embeddings, files


def compute_top_k(query_embeddings, query_files, ref_embeddings, ref_files, k_values):
    """
    对每个query图片，查找top-k个最相似的reference图片
    返回每个query的top-k结果
    """
    results = {k: [] for k in k_values}

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


def evaluate_retrieval(results, query_subclass, k_values):
    """
    评估检索结果
    对于business_cards，假设query图片和对应reference图片有相同的编号（如001.jpg对应001.jpg）
    """
    print(f"\n{'='*60}")
    print(f"Query subclass: {query_subclass}")
    print(f"{'='*60}")

    for k in k_values:
        # 正确的匹配：query图片的文件名（如001.jpg）应该在top-k结果中
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

        accuracy = correct / total if total > 0 else 0
        print(f"Top-{k} Accuracy: {correct}/{total} = {accuracy:.4f}")

    return results


def main():
    # 加载reference embeddings
    print("Loading reference embeddings...")
    ref_embeddings, ref_files = load_embeddings(REF_SUBCLASS)
    print(f"Reference embeddings shape: {ref_embeddings.shape}")
    print(f"Reference files count: {len(ref_files)}")

    # 对每个query子类进行检索
    all_results = {}

    for query_subclass in QUERY_SUBCLASES:
        print(f"\nProcessing query subclass: {query_subclass}")

        # 加载query embeddings
        query_embeddings, query_files = load_embeddings(query_subclass)
        print(f"Query embeddings shape: {query_embeddings.shape}")
        print(f"Query files count: {len(query_files)}")

        # 计算top-k检索结果
        results = compute_top_k(query_embeddings, query_files, ref_embeddings, ref_files, K_VALUES)

        # 评估结果
        evaluate_retrieval(results, query_subclass, K_VALUES)

        all_results[query_subclass] = results

    # 汇总所有子类的结果
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for k in K_VALUES:
        total_correct = 0
        total_count = 0

        for query_subclass in QUERY_SUBCLASES:
            results = all_results[query_subclass][k]
            for result in results:
                query_name = result['query']
                query_base = os.path.splitext(query_name)[0]

                for ref_file in result['ref_files']:
                    ref_base = os.path.splitext(ref_file)[0]
                    if query_base == ref_base:
                        total_correct += 1
                        break
                total_count += 1

        overall_accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"Overall Top-{k} Accuracy: {total_correct}/{total_count} = {overall_accuracy:.4f}")

    print("\nRetrieval evaluation completed!")


if __name__ == "__main__":
    main()