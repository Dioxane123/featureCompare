"""
定性可视化：展示DINOv3和CLIP在SMVS和DISC21数据集上的检索成功/失败案例。
直接从已有的result文件夹加载特征，无需重新提取。
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib
matplotlib.use('Agg')

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']

# ===================== 配置 =====================
# 选取代表性模型：各选一个base规模进行对比
DINO_MODEL = 'vit_large_patch16_dinov3.lvd1689m'
CLIP_MODEL = 'vit_large_patch14_clip_224.openai'

SMVS_ROOT = '/media/dioxane/MovieDisk/Dataset/SMVS'
DISC21_ROOT = '/media/dioxane/MovieDisk/Dataset/disc21'
DISC21_GT_FILE = os.path.join(DISC21_ROOT, 'filter_final_gt.csv')

RESULT_DIR = 'result'
OUTPUT_DIR = 'visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 5  # 展示top-5检索结果
NUM_SUCCESS = 3  # 每个模型展示的成功案例数
NUM_FAILURE = 3  # 每个模型展示的失败案例数

# SMVS中选择几个有代表性的task来可视化
SMVS_VIS_TASKS = ['business_cards', 'cd_covers', 'dvd_covers', 'landmarks', 'museum_paintings', 'book_covers']


def cosine_similarity_batch(query, refs):
    """批量计算余弦相似度"""
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    refs_norm = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-8)
    return refs_norm @ query_norm


def load_image(path, size=(128, 128)):
    """加载并缩放图片"""
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        # 返回灰色占位图
        return np.ones((*size, 3), dtype=np.uint8) * 128


# ===================== SMVS 可视化 =====================

def get_smvs_query_subclasses(task_dir):
    """获取SMVS task的query子类"""
    subclasses = []
    for item in os.listdir(task_dir):
        if item.endswith('_embeddings.npy'):
            name = item.replace('_embeddings.npy', '')
            if name not in ('Reference', 'Query'):
                subclasses.append(name)
    # 如果没有子类，说明是用Query/Reference命名的
    if not subclasses:
        if os.path.exists(os.path.join(task_dir, 'Query_embeddings.npy')):
            subclasses = ['Query']
    return sorted(subclasses)


def smvs_retrieval_for_model(model_name, task):
    """对SMVS中某个task执行检索，返回所有query的检索结果"""
    task_dir = os.path.join(RESULT_DIR, model_name, task)
    if not os.path.exists(task_dir):
        return []

    query_subclasses = get_smvs_query_subclasses(task_dir)

    # 加载reference
    ref_emb_path = os.path.join(task_dir, 'Reference_embeddings.npy')
    if not os.path.exists(ref_emb_path):
        return []
    ref_embeddings = np.load(ref_emb_path)
    ref_files = np.load(os.path.join(task_dir, 'Reference_files.npy'), allow_pickle=True)

    # 确定reference图片路径
    if task in ('landmarks', 'museum_paintings', 'video_frames'):
        ref_img_dir = os.path.join(SMVS_ROOT, task, 'Reference')
    else:
        ref_img_dir = os.path.join(SMVS_ROOT, task, 'Reference')

    all_results = []
    for subclass in query_subclasses:
        q_emb = np.load(os.path.join(task_dir, f'{subclass}_embeddings.npy'))
        q_files = np.load(os.path.join(task_dir, f'{subclass}_files.npy'), allow_pickle=True)

        # 确定query图片路径
        if subclass == 'Query':
            q_img_dir = os.path.join(SMVS_ROOT, task, 'Query')
        else:
            q_img_dir = os.path.join(SMVS_ROOT, task, subclass)

        for i in range(len(q_emb)):
            sims = cosine_similarity_batch(q_emb[i], ref_embeddings)
            top_indices = np.argsort(sims)[::-1][:TOP_K]

            query_base = os.path.splitext(q_files[i])[0]
            # 检查top-1是否匹配
            top1_ref_base = os.path.splitext(ref_files[top_indices[0]])[0]
            is_top1_correct = (query_base == top1_ref_base)

            # 检查top-k中是否有匹配
            is_topk_correct = any(
                os.path.splitext(ref_files[idx])[0] == query_base
                for idx in top_indices
            )

            all_results.append({
                'query_file': q_files[i],
                'query_img_path': os.path.join(q_img_dir, q_files[i]),
                'subclass': subclass,
                'top_k_files': [ref_files[idx] for idx in top_indices],
                'top_k_sims': [sims[idx] for idx in top_indices],
                'top_k_img_paths': [os.path.join(ref_img_dir, ref_files[idx]) for idx in top_indices],
                'is_top1_correct': is_top1_correct,
                'is_topk_correct': is_topk_correct,
                'correct_ref_base': query_base,
            })

    return all_results


def disc21_load_ground_truth():
    """加载DISC21 ground truth"""
    gt = {}
    with open(DISC21_GT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    gt[parts[0]] = parts[1]
    return gt


def disc21_retrieval_for_model(model_name):
    """对DISC21执行检索"""
    disc_dir = os.path.join(RESULT_DIR, model_name, 'DISC21')
    if not os.path.exists(disc_dir):
        return []

    q_emb = np.load(os.path.join(disc_dir, 'query_embeddings.npy'))
    q_files = np.load(os.path.join(disc_dir, 'query_files.npy'), allow_pickle=True)
    r_emb = np.load(os.path.join(disc_dir, 'reference_embeddings.npy'))
    r_files = np.load(os.path.join(disc_dir, 'reference_files.npy'), allow_pickle=True)

    gt = disc21_load_ground_truth()

    # query图片路径
    q_img_dir = os.path.join(DISC21_ROOT, 'final_queries', 'images', 'final_queries')
    r_img_dir = os.path.join(DISC21_ROOT, 'references_0', 'images', 'references')

    all_results = []
    for i in range(len(q_emb)):
        sims = cosine_similarity_batch(q_emb[i], r_emb)
        top_indices = np.argsort(sims)[::-1][:TOP_K]

        query_id = q_files[i]  # e.g. 'Q50060'
        correct_ref_id = gt.get(query_id)

        if correct_ref_id is None:
            continue

        # 检查top-k结果
        top_k_ref_ids = [r_files[idx] for idx in top_indices]
        is_top1_correct = (top_k_ref_ids[0] == correct_ref_id)
        is_topk_correct = correct_ref_id in top_k_ref_ids

        all_results.append({
            'query_file': query_id,
            'query_img_path': os.path.join(q_img_dir, query_id + '.jpg'),
            'top_k_files': top_k_ref_ids,
            'top_k_sims': [sims[idx] for idx in top_indices],
            'top_k_img_paths': [os.path.join(r_img_dir, rid + '.jpg') for rid in top_k_ref_ids],
            'is_top1_correct': is_top1_correct,
            'is_topk_correct': is_topk_correct,
            'correct_ref_id': correct_ref_id,
            'correct_ref_img_path': os.path.join(r_img_dir, correct_ref_id + '.jpg'),
        })

    return all_results


def pick_cases(results, num_success, num_failure):
    """从检索结果中挑选成功和失败案例"""
    successes = [r for r in results if r['is_top1_correct']]
    failures = [r for r in results if not r['is_top1_correct']]

    # 成功案例按相似度从高到低排，选最有信心的
    successes.sort(key=lambda x: x['top_k_sims'][0], reverse=True)
    # 失败案例按top1相似度从高到低排，选最"自信但错误"的
    failures.sort(key=lambda x: x['top_k_sims'][0], reverse=True)

    picked_success = successes[:num_success]
    picked_failure = failures[:num_failure]

    return picked_success, picked_failure


def draw_retrieval_figure(cases, title, save_path, show_correct_ref=False):
    """
    绘制检索结果可视化图。
    每行: Query图片 + Top-K检索结果，正确的用绿色边框，错误的用红色边框。
    """
    if not cases:
        print(f"  No cases to draw for: {title}")
        return

    n_rows = len(cases)
    n_cols = 1 + TOP_K + (1 if show_correct_ref else 0)
    img_size = 128

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    for row_idx, case in enumerate(cases):
        ax_row = axes[row_idx]

        # 绘制Query
        ax = ax_row[0]
        q_img = load_image(case['query_img_path'])
        ax.imshow(q_img)
        ax.set_title(f"Query\n{case['query_file']}", fontsize=7)
        ax.axis('off')
        # 蓝色边框
        rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                          edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

        # 绘制Top-K
        for k in range(TOP_K):
            ax = ax_row[1 + k]
            r_img = load_image(case['top_k_img_paths'][k])
            ax.imshow(r_img)

            ref_file = case['top_k_files'][k]
            sim_val = case['top_k_sims'][k]

            # 判断是否为正确匹配
            if 'correct_ref_base' in case:
                ref_base = os.path.splitext(ref_file)[0]
                is_match = (ref_base == case['correct_ref_base'])
            else:
                is_match = (ref_file == case.get('correct_ref_id'))

            color = 'green' if is_match else 'red'
            ax.set_title(f"Top-{k+1} ({sim_val:.3f})\n{ref_file}", fontsize=6)
            ax.axis('off')
            rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                              edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # 如果需要，绘制正确的reference
        if show_correct_ref and 'correct_ref_img_path' in case:
            ax = ax_row[-1]
            gt_img = load_image(case['correct_ref_img_path'])
            ax.imshow(gt_img)
            ax.set_title(f"GT\n{case.get('correct_ref_id', '')}", fontsize=7)
            ax.axis('off')
            rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                              edgecolor='gold', facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_smvs():
    """SMVS数据集可视化"""
    print("\n" + "=" * 60)
    print("SMVS Qualitative Visualization")
    print("=" * 60)

    smvs_out = os.path.join(OUTPUT_DIR, 'SMVS')
    os.makedirs(smvs_out, exist_ok=True)

    for task in SMVS_VIS_TASKS:
        print(f"\n--- Task: {task} ---")

        for model_name, model_label in [(DINO_MODEL, 'DINOv3'), (CLIP_MODEL, 'CLIP')]:
            print(f"  Model: {model_label} ({model_name})")
            results = smvs_retrieval_for_model(model_name, task)

            if not results:
                print(f"    No results found, skipping.")
                continue

            n_correct = sum(1 for r in results if r['is_top1_correct'])
            print(f"    Total queries: {len(results)}, Top-1 correct: {n_correct} ({n_correct/len(results)*100:.1f}%)")

            successes, failures = pick_cases(results, NUM_SUCCESS, NUM_FAILURE)

            if successes:
                draw_retrieval_figure(
                    successes,
                    f"SMVS/{task} - {model_label} - Success Cases (Top-1 Correct)",
                    os.path.join(smvs_out, f'{task}_{model_label}_success.png')
                )

            if failures:
                draw_retrieval_figure(
                    failures,
                    f"SMVS/{task} - {model_label} - Failure Cases (Top-1 Incorrect)",
                    os.path.join(smvs_out, f'{task}_{model_label}_failure.png')
                )

    # 绘制对比图：同一query在两个模型下的检索对比
    print(f"\n--- Generating DINOv3 vs CLIP comparison ---")
    compare_out = os.path.join(smvs_out, 'comparison')
    os.makedirs(compare_out, exist_ok=True)

    for task in SMVS_VIS_TASKS:
        dino_results = smvs_retrieval_for_model(DINO_MODEL, task)
        clip_results = smvs_retrieval_for_model(CLIP_MODEL, task)

        if not dino_results or not clip_results:
            continue

        # 建立query索引
        dino_dict = {r['query_file']: r for r in dino_results}
        clip_dict = {r['query_file']: r for r in clip_results}
        common_queries = set(dino_dict.keys()) & set(clip_dict.keys())

        # 找DINOv3成功但CLIP失败的案例
        dino_wins = [q for q in common_queries
                     if dino_dict[q]['is_top1_correct'] and not clip_dict[q]['is_top1_correct']]
        # 找CLIP成功但DINOv3失败的案例
        clip_wins = [q for q in common_queries
                     if clip_dict[q]['is_top1_correct'] and not dino_dict[q]['is_top1_correct']]

        if dino_wins or clip_wins:
            draw_comparison_figure(
                dino_dict, clip_dict, dino_wins[:3], clip_wins[:3],
                task, os.path.join(compare_out, f'{task}_comparison.png')
            )


def draw_comparison_figure(dino_dict, clip_dict, dino_wins, clip_wins, task, save_path):
    """绘制DINOv3 vs CLIP对比图"""
    cases = []
    labels = []

    for q in dino_wins:
        cases.append(('DINOv3 wins', dino_dict[q], clip_dict[q]))
    for q in clip_wins:
        cases.append(('CLIP wins', dino_dict[q], clip_dict[q]))

    if not cases:
        return

    n_rows = len(cases)
    # 每行: Query | DINOv3 Top-3 | CLIP Top-3
    n_cols = 1 + 3 + 3  # query + 3 dino + 3 clip
    img_size = 128

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(f"SMVS/{task} - DINOv3 vs CLIP Comparison", fontsize=13, fontweight='bold', y=1.02)

    # 列标题
    col_labels = ['Query', 'DINOv3\nTop-1', 'DINOv3\nTop-2', 'DINOv3\nTop-3',
                  'CLIP\nTop-1', 'CLIP\nTop-2', 'CLIP\nTop-3']

    for row_idx, (label, dino_case, clip_case) in enumerate(cases):
        ax_row = axes[row_idx]

        # Query
        ax = ax_row[0]
        q_img = load_image(dino_case['query_img_path'])
        ax.imshow(q_img)
        title_str = f"{label}\n{dino_case['query_file']}" if row_idx == 0 or True else dino_case['query_file']
        ax.set_title(title_str, fontsize=7, color='darkblue' if 'DINOv3' in label else 'darkred')
        ax.axis('off')
        rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                          edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

        query_base = dino_case['correct_ref_base']

        # DINOv3 Top-3
        for k in range(3):
            ax = ax_row[1 + k]
            r_img = load_image(dino_case['top_k_img_paths'][k])
            ax.imshow(r_img)
            ref_base = os.path.splitext(dino_case['top_k_files'][k])[0]
            is_match = (ref_base == query_base)
            color = 'green' if is_match else 'red'
            ax.set_title(f"{dino_case['top_k_sims'][k]:.3f}", fontsize=7)
            ax.axis('off')
            rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                              edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if row_idx == 0:
                ax.text(0.5, 1.25, col_labels[1 + k], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        # CLIP Top-3
        for k in range(3):
            ax = ax_row[4 + k]
            r_img = load_image(clip_case['top_k_img_paths'][k])
            ax.imshow(r_img)
            ref_base = os.path.splitext(clip_case['top_k_files'][k])[0]
            is_match = (ref_base == query_base)
            color = 'green' if is_match else 'red'
            ax.set_title(f"{clip_case['top_k_sims'][k]:.3f}", fontsize=7)
            ax.axis('off')
            rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                              edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if row_idx == 0:
                ax.text(0.5, 1.25, col_labels[4 + k], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {save_path}")


def visualize_disc21():
    """DISC21数据集可视化"""
    print("\n" + "=" * 60)
    print("DISC21 Qualitative Visualization")
    print("=" * 60)

    disc_out = os.path.join(OUTPUT_DIR, 'DISC21')
    os.makedirs(disc_out, exist_ok=True)

    for model_name, model_label in [(DINO_MODEL, 'DINOv3'), (CLIP_MODEL, 'CLIP')]:
        print(f"\n  Model: {model_label} ({model_name})")
        results = disc21_retrieval_for_model(model_name)

        if not results:
            print(f"    No results found, skipping.")
            continue

        n_correct = sum(1 for r in results if r['is_top1_correct'])
        print(f"    Total queries: {len(results)}, Top-1 correct: {n_correct} ({n_correct/len(results)*100:.1f}%)")

        successes, failures = pick_cases(results, NUM_SUCCESS, NUM_FAILURE)

        if successes:
            draw_retrieval_figure(
                successes,
                f"DISC21 - {model_label} - Success Cases (Top-1 Correct)",
                os.path.join(disc_out, f'{model_label}_success.png'),
                show_correct_ref=False
            )

        if failures:
            draw_retrieval_figure(
                failures,
                f"DISC21 - {model_label} - Failure Cases (Top-1 Incorrect)",
                os.path.join(disc_out, f'{model_label}_failure.png'),
                show_correct_ref=True
            )

    # DINOv3 vs CLIP 对比
    print(f"\n--- Generating DINOv3 vs CLIP comparison for DISC21 ---")
    dino_results = disc21_retrieval_for_model(DINO_MODEL)
    clip_results = disc21_retrieval_for_model(CLIP_MODEL)

    if dino_results and clip_results:
        dino_dict = {r['query_file']: r for r in dino_results}
        clip_dict = {r['query_file']: r for r in clip_results}
        common_queries = set(dino_dict.keys()) & set(clip_dict.keys())

        dino_wins = [q for q in common_queries
                     if dino_dict[q]['is_top1_correct'] and not clip_dict[q]['is_top1_correct']]
        clip_wins = [q for q in common_queries
                     if clip_dict[q]['is_top1_correct'] and not dino_dict[q]['is_top1_correct']]

        if dino_wins or clip_wins:
            draw_disc21_comparison(
                dino_dict, clip_dict, sorted(dino_wins)[:3], sorted(clip_wins)[:3],
                os.path.join(disc_out, 'comparison.png')
            )


def draw_disc21_comparison(dino_dict, clip_dict, dino_wins, clip_wins, save_path):
    """绘制DISC21的DINOv3 vs CLIP对比图"""
    cases = []
    for q in dino_wins:
        cases.append(('DINOv3 wins', dino_dict[q], clip_dict[q]))
    for q in clip_wins:
        cases.append(('CLIP wins', dino_dict[q], clip_dict[q]))

    if not cases:
        return

    n_rows = len(cases)
    n_cols = 1 + 3 + 3 + 1  # query + dino top3 + clip top3 + GT
    img_size = 128

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle("DISC21 - DINOv3 vs CLIP Comparison", fontsize=13, fontweight='bold', y=1.02)

    for row_idx, (label, dino_case, clip_case) in enumerate(cases):
        ax_row = axes[row_idx]

        # Query
        ax = ax_row[0]
        q_img = load_image(dino_case['query_img_path'])
        ax.imshow(q_img)
        ax.set_title(f"{label}\n{dino_case['query_file']}", fontsize=7,
                     color='darkblue' if 'DINOv3' in label else 'darkred')
        ax.axis('off')
        rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                          edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

        correct_ref_id = dino_case['correct_ref_id']

        # DINOv3 Top-3
        for k in range(3):
            ax = ax_row[1 + k]
            r_img = load_image(dino_case['top_k_img_paths'][k])
            ax.imshow(r_img)
            is_match = (dino_case['top_k_files'][k] == correct_ref_id)
            color = 'green' if is_match else 'red'
            sim_str = f"DINOv3 Top-{k+1}\n{dino_case['top_k_sims'][k]:.3f}"
            ax.set_title(sim_str, fontsize=6)
            ax.axis('off')
            rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                              edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # CLIP Top-3
        for k in range(3):
            ax = ax_row[4 + k]
            r_img = load_image(clip_case['top_k_img_paths'][k])
            ax.imshow(r_img)
            is_match = (clip_case['top_k_files'][k] == correct_ref_id)
            color = 'green' if is_match else 'red'
            sim_str = f"CLIP Top-{k+1}\n{clip_case['top_k_sims'][k]:.3f}"
            ax.set_title(sim_str, fontsize=6)
            ax.axis('off')
            rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                              edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # GT
        ax = ax_row[7]
        gt_img = load_image(dino_case['correct_ref_img_path'])
        ax.imshow(gt_img)
        ax.set_title(f"GT\n{correct_ref_id}", fontsize=7)
        ax.axis('off')
        rect = Rectangle((0, 0), img_size - 1, img_size - 1, linewidth=3,
                          edgecolor='gold', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {save_path}")


if __name__ == '__main__':
    visualize_smvs()
    visualize_disc21()
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
