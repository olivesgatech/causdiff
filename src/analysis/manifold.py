import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_manifold_structure(sub_intentions, global_intentions):
    """
    같은 global intention은 같은 manifold, 다른 intention은 다른 manifold인지 분석
    """
    S, T, D = sub_intentions.shape
    
    # 각 시간 τ별로 분석
    for τ in [0, T//2, T-1]:  # 초기, 중간, 마지막 시간
        print(f"\n=== 시간 τ={τ} 분석 ===")
        
        intention_labels = []
        all_samples = []
        
        samples = sub_intentions[:, τ, :]  # (S, D)
        all_samples.extend(samples)
        intention_labels.extend([global_intentions] * S)
    return all_samples, intention_labels

def visualize_manifold_structure(sub_intentions, intention_labels):
    
    intention_labels = np.array(intention_labels)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    samples_2d = tsne.fit_transform(sub_intentions)
    
    # 2x2 그리드로 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Intention별 색상
    unique_intentions = list(set(intention_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_intentions)))
    color_map = {intent: colors[i] for i, intent in enumerate(unique_intentions)}
    
    for i, intention in enumerate(unique_intentions):
        mask = [l == intention for l in intention_labels]
        axes[0,0].scatter(samples_2d[mask, 0], samples_2d[mask, 1], 
                         c=[color_map[intention]], label=intention, alpha=0.6, s=10)
    axes[0,0].set_title(f'Intention별 분포 ({len(sub_intentions)}개 샘플)')
    axes[0,0].legend()
    
    # 2. 시간별 색상 (전체)
    scatter = axes[0,1].scatter(samples_2d[:, 0], samples_2d[:, 1], 
                               c=time_labels, cmap='viridis', alpha=0.6, s=10)
    axes[0,1].set_title('시간별 분포 (τ)')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # 3. 각 intention별 시간에 따른 변화
    for idx, intention in enumerate(unique_intentions[:4]):  # 처음 4개만
        row, col = 1 + idx//2, idx % 2
        if row < 2 and col < 2:
            mask = [l == intention for l in intention_labels]
            intention_samples = samples_2d[mask]
            intention_times = [time_labels[i] for i in range(len(time_labels)) if mask[i]]
            
            scatter = axes[row, col].scatter(intention_samples[:, 0], intention_samples[:, 1], 
                                           c=intention_times, cmap='plasma', alpha=0.7, s=15)
            axes[row, col].set_title(f'{intention} - 시간별 분포')
            plt.colorbar(scatter, ax=axes[row, col])
    
    plt.tight_layout()
    plt.savefig('simple_comprehensive_manifold.png', dpi=300, bbox_inches='tight')
    plt.close()

    return samples_2d, intention_labels, time_labels