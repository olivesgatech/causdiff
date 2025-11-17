import numpy as np
import os
from scipy.linalg import sqrtm

def compute_intrinsic_uncertainty(sub_intentions):
    """
    sub-intention의 내재적 불확실성 계산
    각 uncertainty 메트릭을 따로 반환
    """
    S, T, D = sub_intentions.shape
    
    # 다양한 uncertainty 메트릭
    results = {
        'variance': np.zeros((T)),
        'consistency': np.zeros((T)),
        'predictive_entropy': np.zeros((T))
    }
    for τ in range(T):
        samples = sub_intentions[:, τ, :]  # (S, D)
        
        # 1. 기본 분산
        results['variance'][τ] = np.mean(np.var(samples, axis=0))
        
        # 2. 샘플 간 일관성 (코사인 유사도 기반)
        if S > 1:
            # 샘플들 간의 평균 코사인 유사도
            norms = np.linalg.norm(samples, axis=1, keepdims=True)
            normalized = samples / (norms + 1e-8)
            cosine_sim = np.dot(normalized, normalized.T)
            avg_similarity = (np.sum(cosine_sim) - S) / (S * (S - 1))  # 대각선 제외
            results['consistency'][τ] = 1 - avg_similarity  # 불일치도
        else:
            results['consistency'][τ] = 0
        
        # 3. 예측 엔트로피 (주성분 분석 기반)
        if S > 1:
            cov_matrix = np.cov(samples.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-8]
            if len(eigenvalues) > 0:
                normalized_eig = eigenvalues / np.sum(eigenvalues)
                entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-8))
                results['predictive_entropy'][τ] = entropy / np.log(D)  # 정규화
            else:
                results['predictive_entropy'][τ] = 0
    
    return results

def load_visual_features(file_list, base_path):
    """
    visual features를 불러오는 함수
    각 비디오마다 길이가 달라서 리스트 형태로 반환
    """
    visual_features = []
    video_lengths = []
    
    for filename in file_list:
        # .txt를 _1.npy로 replace
        npy_filename = filename.replace('.txt', '_1.npy')
        npy_path = os.path.join(base_path, npy_filename)
        
        if os.path.exists(npy_path):
            feature = np.load(npy_path)
            length = int(len(feature) * 0.7)
            feature = feature[:length]
            feature = feature[::15]
            visual_features.append(feature)
            video_lengths.append(feature.shape[0])
            print(f"Loaded: {npy_filename} - shape: {feature.shape}")
        else:
            print(f"Warning: File not found - {npy_path}")
    
    return visual_features


def analyze_uncertainty_correlations(uncertainty_metrics, visual_features):
    """
    각 uncertainty 메트릭과 시각 정보의 연관성 분석
    """
    T = uncertainty_metrics['variance'].shape
    
    print("=== Uncertainty 메트릭과 시각 정보의 연관성 분석 ===")
    
    # 시각 정보의 변화량 계산
    visual_changes = np.linalg.norm(np.diff(visual_features, axis=0), axis=1)
    visual_changes = np.concatenate([[0], visual_changes])  # 첫 프레임 padding
    
    # 각 uncertainty 메트릭과의 상관관계 계산
    for metric_name, uncertainty in uncertainty_metrics.items():
        correlation = np.corrcoef(uncertainty, visual_changes)[0, 1]
        
        print(f"  {metric_name}:")
        print(f"    - 시각 변화 상관관계: {correlation:.3f}")
        
        # 상관관계 해석
        if abs(correlation) > 0.3:
            direction = "증가" if correlation > 0 else "감소"
            print(f"    - 해석: 시각 변화가 클수록 {metric_name} {direction}")
        else:
            print(f"    - 해석: 시각 변화와 {metric_name} 관계 미미")
    
    # 메트릭들 간의 상관관계
    print(f"  메트릭 간 상관관계:")
    var_cons = np.corrcoef(uncertainty_metrics['variance'], 
                            uncertainty_metrics['consistency'])[0, 1]
    var_ent = np.corrcoef(uncertainty_metrics['variance'], 
                            uncertainty_metrics['predictive_entropy'])[0, 1]
    cons_ent = np.corrcoef(uncertainty_metrics['consistency'], 
                            uncertainty_metrics['predictive_entropy'])[0, 1]
    
    print(f"    - 분산-불일치도: {var_cons:.3f}")
    print(f"    - 분산-엔트로피: {var_ent:.3f}")
    print(f"    - 불일치도-엔트로피: {cons_ent:.3f}")

def plot_uncertainty_analysis(uncertainty_metrics, visual_features, out_prefix=''):
    """
    uncertainty 메트릭과 시각 정보를 시각화
    """
    import matplotlib.pyplot as plt
    
    T = uncertainty_metrics['variance'].shape[0]
    t_length = int(T*0.2/0.7)
    
    plt.figure(figsize=(15, 12))
    
    # 1. 각 uncertainty 메트릭
    plt.subplot(3, 1, 1)
    plt.plot(uncertainty_metrics['variance'][:t_length], 'r-', label='variance', alpha=0.8)
    #plt.plot(uncertainty_metrics['consistency']/ np.max(uncertainty_metrics['consistency']), 'g-', label='inconsistency', alpha=0.8)
    #plt.plot(uncertainty_metrics['predictive_entropy'] / np.max(uncertainty_metrics['predictive_entropy']), 'b-', label='predictive entropy', alpha=0.8)
    #plt.plot(uncertainty_metrics['u_vn'], 'o-', label='u-vn', alpha=0.8)
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty metric comparisons')
    plt.legend()
    plt.grid(True)
    
    # 시각정보 변화량
    plt.subplot(3, 1, 2)
    visual_features = visual_features[:t_length]
    visual_changes = np.linalg.norm(np.diff(visual_features, axis=0), axis=1)
    visual_changes = np.concatenate([[0], visual_changes])
    plt.plot(visual_changes, 'm-', label='visual info changes')
    plt.ylabel('changes')
    plt.title('visual info changes')
    plt.grid(True)
    plt.legend()
    
    # 3. 정규화된 비교
    plt.subplot(3, 1, 3)
    # 각 메트릭 정규화
    var_norm = uncertainty_metrics['variance'] / np.max(uncertainty_metrics['variance'])
    cons_norm = uncertainty_metrics['consistency'] / np.max(uncertainty_metrics['consistency'])
    ent_norm = uncertainty_metrics['predictive_entropy'] / np.max(uncertainty_metrics['predictive_entropy'])
    visual_norm = visual_changes / np.max(visual_changes)
    
    plt.plot(var_norm, 'r--', label='variance', alpha=0.7)
    plt.plot(cons_norm, 'g--', label='inconsistency', alpha=0.7)
    plt.plot(ent_norm, 'b--', label='entropy', alpha=0.7)
    plt.plot(visual_norm, 'm-', label='visual info variance', alpha=0.7)
    plt.xlabel('video time τ')
    plt.ylabel('normalized')
    plt.title('Uncertainty metric and visual info variance comparison')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    fname = f"{out_prefix}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()

def compute_visual_complexity(visual_feature):
    """
    시각 정보의 복잡성/정보량 계산 (엔트로피 기반)
    """
    # 각 프레임의 엔트로피 계산
    complexities = []
    for t in range(visual_feature.shape[0]):
        # 특징 벡터의 분포 엔트로피
        hist, _ = np.histogram(visual_feature[t], bins=20, density=True)
        hist = hist / hist.sum()
        entropy_val = -np.sum(hist * np.log(hist + 1e-8))
        complexities.append(entropy_val)
    
    return np.array(complexities)