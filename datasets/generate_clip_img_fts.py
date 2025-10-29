import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import re
from PIL import Image

# --- NEW: CLIP (open_clip) instead of ResNet ---
import open_clip

# Choose CLIP model + weights
CLIP_MODEL_NAME = "ViT-B-32"      # e.g., "ViT-B-32", "ViT-L-14"
CLIP_PRETRAINED = "openai"        # e.g., "openai", "laion2b_s32b_b79k"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create model & preprocess
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
)
clip_model.eval()

# actions_dict와 query_dict를 로드하는 함수 (원본 그대로)
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, label = line.strip().split()
            mapping[label] = int(idx)
    return mapping

class BaseDataset(Dataset):
    def __init__(self, train_split_path, actions_dict, query_dict, pad_idx, n_class, args=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.query_dict = query_dict
        self.pad_idx = pad_idx

        self.args = args

        # Load specific files listed in train_split.txt
        with open(train_split_path, 'r') as f:
            self.vid_list = [os.path.join(args.groundtruth_dir, line.strip()) for line in f]

    def preprocess_image(self, image_path):
        """
        Load an image and return a CLIP image embedding with shape [1, D].
        Uses CLIP's official preprocess pipeline; no manual cv2 resize/normalize.
        """
        try:
            # Prefer PIL for robust decoding; convert to RGB explicitly
            img = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback: try cv2 then convert to PIL if needed
            try:
                bgr = cv2.imread(image_path)
                if bgr is None:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
            except Exception:
                return None

        # Apply CLIP transforms
        img_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        with torch.no_grad():
            feats = clip_model.encode_image(img_tensor)             # [1, D]
            # L2-normalize (standard for CLIP retrieval)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            feats = feats.float().cpu()

        return feats  # shape: [1, D]

    def save_preprocessed_sequences(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 비디오 파일마다 처리
        for gt_file in self.vid_list:
            base_filename = os.path.basename(gt_file).replace('.txt', '')
            existing_files = [f for f in os.listdir(save_dir) if f.startswith(base_filename)]

            # 이미 처리된 파일이 있으면 건너뜀
            if existing_files:
                print(f"Skipping {gt_file} (already processed)")
                continue

            with open(gt_file, 'r') as file_ptr:
                lines = file_ptr.readlines()
                # 유효한 라인만 포함하도록 필터링
                valid_lines = [line.strip() for line in lines]

            features = []
            l2_labels = []
            l3_labels = []
            image_paths = []
            seq_idx = 1  # 시퀀스 번호

            # 유효한 모든 라인에 대해 처리
            for line in valid_lines:
                split_data = line.split(',')
                image_path, l2_label, l3_label = split_data
                image_path = image_path.replace('/home/seulgi/work/', '/media/seulgi/532184bc-f309-4eee-8309-2287e4641237/')

                # 이미지 전처리 -> CLIP 임베딩
                image_tensor = self.preprocess_image(image_path)
                if image_tensor is not None:
                    features.append(image_tensor.numpy())  # append [1, D]
                    l2_labels.append(l2_label)
                    l3_labels.append(l3_label)
                    image_paths.append(image_path)
                else:
                    print(f"Warning: {image_path} cannot load the image. Saving...")
                    if features:
                        self._save_sequence(save_dir, gt_file, seq_idx, features, l2_labels, l3_labels, image_paths)
                    # 새로운 시퀀스 시작
                    features = []
                    l2_labels = []
                    l3_labels = []
                    image_paths = []
                    seq_idx += 1

            # 루프가 종료된 후 남아있는 시퀀스를 저장
            if features:  # features가 비어있지 않은 경우에만 저장
                self._save_sequence(save_dir, gt_file, seq_idx, features, l2_labels, l3_labels, image_paths)

    def _save_sequence(self, save_dir, gt_file, seq_idx, features, l2_labels, l3_labels, image_paths):
        # .npy 파일로 features 저장
        base_filename = os.path.basename(gt_file).replace('.txt', '')
        npy_file_name = f"{base_filename}_{seq_idx}.npy"
        npy_file_path = os.path.join(save_dir, npy_file_name)
        # features is a list of [1, D] arrays -> stack along time -> [T, D]
        np.save(npy_file_path, np.vstack(features))

        # 텍스트 파일에 라벨과 이미지 경로 저장
        txt_file_name = f"{base_filename}_{seq_idx}.txt"
        txt_file_path = os.path.join(save_dir, txt_file_name)

        with open(txt_file_path, 'w') as f:
            for idx in range(len(l2_labels)):
                f.write(f"{image_paths[idx]},{l2_labels[idx]},{l3_labels[idx]}\n")

        print(f"Saved: {txt_file_path} and {npy_file_path}")

def main():
    # 경로 설정
    train_split_path = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/splits/val_split.txt'
    l2_mapping_file = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/mapping_l2.txt'
    l3_mapping_file = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/mapping_l3.txt'
    save_dir = '/home/seulgi/work/causdiff/datasets/darai/features_clip'  # 시퀀스를 저장할 디렉터리 경로

    # actions_dict와 query_dict 로드
    actions_dict = load_mapping(l2_mapping_file)
    query_dict = load_mapping(l3_mapping_file)

    pad_idx = 0
    n_class = len(actions_dict)

    class Args:
        groundtruth_dir = '/home/seulgi/work/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_temp'

    args = Args()

    # BaseDataset 생성 및 시퀀스 저장
    dataset = BaseDataset(train_split_path, actions_dict, query_dict, pad_idx, n_class, args=args)
    dataset.save_preprocessed_sequences(save_dir)

if __name__ == "__main__":
    main()
