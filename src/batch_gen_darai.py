import torch
import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
import re
import glob
from PIL import Image


def npy_to_jpg_path(
    npy_path: str,
    out_root: str = "/home/hice1/skim3513/AIFirst_F24_data/darai/RGB_sd",
    frame_idx: int = 0, file_length=0,   # 기본은 00000.jpg 로 생성
) -> str:
    """
    예시 입력:
      /.../features_img/01_2_camera_1_fps_15_Using handheld smart devices_1.npy
    예시 출력:
      /.../RGB_sd/Using handheld smart devices/camera_1_fps_15/01_2_00000.jpg
    """
    name = Path(npy_path).stem  # 확장자 제거
    # 패턴: [앞번호]_[세그] _camera_[카메라] _fps_[fps] _[활동명(공백 포함)] _[클립/인덱스]
    m = re.match(
        r'^(?P<prefix>\d+_\d+)_camera_(?P<cam>\d+)_fps_(?P<fps>\d+)_(?P<activity>.+)_(?P<tail>\d+)$',
        name
    )
    if not m:
        raise ValueError(f"파일명 형식이 예상과 다릅니다: {name}")

    prefix   = m["prefix"]                         # 예: 01_2
    cam      = m["cam"]                            # 예: 1
    fps      = m["fps"]                            # 예: 15
    activity = m["activity"].strip()               # 예: 'Using handheld smart devices'

    # 출력 경로 조립
    out_dir = Path(out_root) / activity / f"camera_{cam}_fps_{fps}"
    
    out_name = f"{prefix}_{frame_idx:05d}.jpg"     # 예: 01_2_00000.jpg
    return str(out_dir / out_name)

class BatchGeneratorTCN(Dataset):
    def __init__(
        self, mode, actions_dict, goals_dict, sample_rate, vid_list_file, pred_perc, obs_perc, args
    ):
        print(f"Dataset: {args.ds}")
        print(f"Mode : {mode}")
        print(f"Vid list file : {vid_list_file}")
        print(f"Observation perc : {obs_perc}")

        # Set params
        self.mode = mode
        self.num_highlevel_classes = int(args.num_highlevel_classes)
        self.num_classes = args.num_classes
        self.actions_dict = actions_dict
        self.goals_dict = goals_dict
        self.dataset = args.ds

        self.sample_rate = sample_rate
        self.obs_perc = obs_perc
        self.pred_perc = pred_perc
        self.part_obs = args.part_obs

        self.qwen_max_side = 336          # ← Qwen용 리사이즈 한계 (원하면 args로)
        self.frame_stride_for_vlm = self.sample_rate    # ← Qwen 입력 프레임 stride
        self._img_cache = {}              # (선택) 간단 캐시: path -> PIL.Image
        self._img_cache_cap = 2048        # 캐시 상한(너무 크면 메모리 증가)

        # sanity check
        if self.mode != "train":
            assert self.obs_perc != 0

        self.features_path = args.features_path
        self.gt_path = args.gt_path

        # Annotations
        file_ptr = open(vid_list_file, "r")
        list_of_vid = file_ptr.read().split("\n")[:-1]
        
        file_ptr.close()

        self.list_of_examples = list()
        for vid in list_of_vid:
            base_name = os.path.splitext(vid)[0]
            seq_idx = 1
            while True:
                gt_file = os.path.join(self.gt_path, f"{base_name}_{seq_idx}.txt")
                feature_file = os.path.join(self.features_path, f"{base_name}_{seq_idx}.npy")
                rgb_file = npy_to_jpg_path(feature_file)
                
                if os.path.exists(gt_file) and os.path.exists(feature_file) and os.path.exists(rgb_file):
                    with open(gt_file, 'r') as file_ptr:
                        lines = file_ptr.readlines()
                        
                        if obs_perc != 0 and len(lines) * obs_perc > self.sample_rate:
                            self.list_of_examples.append([vid, obs_perc, seq_idx])
                        elif len(lines) * 0.2 > self.sample_rate:
                            assert self.mode == 'train'
                            self.list_of_examples.append([vid, 0.2, seq_idx])
                            self.list_of_examples.append([vid, 0.3, seq_idx])
                            self.list_of_examples.append([vid, 0.5, seq_idx])
                        elif len(lines) * 0.3 > self.sample_rate:
                            self.list_of_examples.append([vid, 0.3, seq_idx])
                            self.list_of_examples.append([vid, 0.5, seq_idx])
                        elif len(lines) * 0.5 > self.sample_rate:
                            self.list_of_examples.append([vid, 0.5, seq_idx])
                        seq_idx += 1
                else:
                    break

    # 파일명 정렬용
    def _natural_key(self, s: str):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    def _expand_first_path_to_sequence(self, first_path: str, frame_stride: int = 15) -> list:
        """
        first_path 예: /.../RGB_sd/Using handheld.../camera_1_fps_15/01_2_00000.jpg
        같은 디렉토리에서 prefix '01_2_'에 해당하는 모든 프레임을 모아
        frame_stride 간격으로 샘플링하여 리스트 반환
        """
        
        d = os.path.dirname(first_path)
        bn = os.path.basename(first_path)              # 01_2_00000.jpg
        parts = bn.split('_')
        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format: {bn}")
        prefix_key = f"{parts[0]}_{parts[1]}_"        # 01_2_

        # 프레임 전부 모으고 자연 정렬
        all_paths = sorted(glob.glob(os.path.join(d, f"{prefix_key}*.jpg")), key=self._natural_key)
        
        if not all_paths:
            raise FileNotFoundError(f"No frames for prefix {prefix_key} under {d}")

        # stride 샘플링
        seq = all_paths[0::max(1, frame_stride)]
        return seq

    def _resize_max_side(self, img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.BICUBIC)

    def _load_resized_image(self, path: str, max_side: int) -> Image.Image:
        # 가벼운 캐시 (선택)
        if path in self._img_cache:
            return self._img_cache[path]
        img = Image.open(path).convert("RGB")
        img = self._resize_max_side(img, max_side)
        if len(self._img_cache) >= self._img_cache_cap:
            # FIFO 식으로 하나 제거
            self._img_cache.pop(next(iter(self._img_cache)))
        self._img_cache[path] = img
        return img

    def _concat_pairs(self, paths: list) -> list:
        """
        연속된 2장 (t, t+1)으로 페어를 만든다.
        길이를 T로 유지하기 위해 마지막은 (last, last)로 중복.
        """
        if len(paths) == 0:
            return []
        pairs = []
        for i in range(len(paths)):
            j = i + 1 if (i + 1) < len(paths) else i
            pairs.append((paths[i], paths[j]))
        return pairs

    def _load_concat_pair(self, p1: str, p2: str, max_side: int) -> Image.Image:
        """
        두 이미지를 로드 → 동일 높이로 맞춤 → 가로로 concat → 최종적으로 max_side 제약으로 리사이즈
        """
        # 개별 프레임을 우선 캐시 로더로 가져와서 (너무 큰 경우) 1차 축소
        img1 = self._load_resized_image(p1, max_side)
        img2 = self._load_resized_image(p2, max_side)

        # 동일 높이로 맞춤 (더 작은 높이에 맞추는 방식)
        h_target = min(img1.height, img2.height)
        if img1.height != h_target:
            new_w1 = int(img1.width * (h_target / img1.height))
            img1 = img1.resize((new_w1, h_target), Image.BICUBIC)
        if img2.height != h_target:
            new_w2 = int(img2.width * (h_target / img2.height))
            img2 = img2.resize((new_w2, h_target), Image.BICUBIC)

        # 가로로 합치기
        w_total = img1.width + img2.width
        concat_img = Image.new("RGB", (w_total, h_target))
        concat_img.paste(img1, (0, 0))
        concat_img.paste(img2, (img1.width, 0))

        # 최종적으로 max_side 제약 확인 후 리사이즈
        concat_img = self._resize_max_side(concat_img, max_side)
        return concat_img


    def __len__(self):
        return len(self.list_of_examples)


    def label_to_id(self, content):
        classes = np.zeros(len(content))
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i].replace(' ', '')]
        return classes

    def goal_label_to_id(self, content):
        classes = self.goals_dict[content.replace(' ', '_')]
        return np.array([classes])


    def __getitem__(self, idx):
        # Load feats and anns
        _, _, seq_idx = self.list_of_examples[idx]
        file_name = self.list_of_examples[idx][0].split(".")[0]+f'_{seq_idx}'
        
        features_name = os.path.join(self.features_path, file_name + ".npy")
        
        #gt_name = os.path.join(self.gt_path, self.list_of_examples[idx][0]+f'_{seq_idx}')
        gt_name = os.path.join(self.gt_path, file_name + ".txt")
        L1 = gt_name.split('_')[-1].split('.')[0]
        
        goals_label = (os.path.basename(gt_name)).split('_')[-2].split('.')[0]
        
        assert os.path.exists(features_name), "{} features file not found".format(features_name)
        assert os.path.exists(gt_name), "{} annot. file not found".format(gt_name)

        features = np.load(features_name).T  # D x T
        file_length = features.shape[1]
        file_ptr = open(gt_name, "r")
        content = file_ptr.read().split("\n")[:-1]
        vid_len = len(content)
        #content = [line.split(',')[1] for line in content]
        content = [line.split(',')[2] for line in content]
        #content = [L1]*vid_len

        # Obs limit
        obs_percentage = self.list_of_examples[idx][1]
        if self.obs_perc == 0:
            assert self.mode == "train"
            # rand aug
            if np.random.random() < 0.4:
                obs_percentage = 0.15 + 0.25 * np.random.random()
        else:
            assert obs_percentage in [0.2, 0.3]
        obs_lim = int(obs_percentage * len(content))

        # Pred limit
        pred_percentage = 1.0 - obs_percentage
        if self.part_obs:
            pred_percentage = 0.5
        pred_lim = int((obs_percentage + pred_percentage) * len(content))
        assert pred_lim <= len(content)


        """ ANNOTATIONS """
        # all labels
        content_past_future = self.label_to_id(content)
        content_past_future = content_past_future[:pred_lim]

        # goal label
        goal_label = self.goal_label_to_id(goals_label)
        
        # masks
        mask_past = np.zeros(len(content_past_future))
        mask_past[:obs_lim] = 1
        mask_past = mask_past[:: self.sample_rate]

        mask_future = np.zeros(len(content_past_future))
        mask_future[obs_lim:] = 1
        mask_future = mask_future[:: self.sample_rate]

        # classes (one hot)
        classes_one_hot = np.zeros((self.num_classes, len(content_past_future)))  # C x T
        for i in range(len(content_past_future)):
            classes_one_hot[int(content_past_future[i])][i] = 1
        classes_one_hot = classes_one_hot[:, :: self.sample_rate]

        # goal (one hot)
        goal_label_extend = goal_label.repeat(len(content_past_future))
        goals_one_hot = np.zeros((int(self.num_highlevel_classes), len(content_past_future)))
        for i in range(len(content_past_future)):
            goals_one_hot[int(goal_label_extend[i])][i] = 1
        goals_one_hot = goals_one_hot[:, :: self.sample_rate]


        """ FEATURES """
        # features
        features_past = features[:, :obs_lim]
        features_past = features_past[:, ::self.sample_rate]

        # all classes
        content = self.label_to_id(content)
        assert len(content) == vid_len
        content_past_future = content_past_future[::self.sample_rate]

        T = classes_one_hot.shape[1]
        first_jpg_path = npy_to_jpg_path(features_name, file_length=obs_lim)  # 기존 그대로 사용
        seq_paths = self._expand_first_path_to_sequence(first_jpg_path, frame_stride=self.sample_rate)

        if len(seq_paths) >= T:
            seq_paths = seq_paths[:T]
        else:
            if len(seq_paths) == 0:
                raise RuntimeError("No frames found to pad from.")
            pad_last = seq_paths[-1]
            seq_paths = seq_paths + [pad_last] * (T - len(seq_paths))

        #pair_paths = self._concat_pairs(seq_paths)
        #image_seq = [self._load_concat_pair(p1, p2, self.qwen_max_side) for (p1, p2) in pair_paths]

        image_seq = [self._load_resized_image(p, self.qwen_max_side) for p in seq_paths]

        sample = {
            "features": features_past,
            "classes": content_past_future,
            "classes_all": content,
            "classes_one_hot": classes_one_hot,
            "mask_past": mask_past,
            "mask_future": mask_future,
            "vid_len": vid_len,
            "file_name": file_name,
            "goal":goal_label,
            "goal_one_hot": goals_one_hot,
            "video_file": image_seq,
        }
        return sample

    def custom_collate(self, batch):
        """COLLECT"""
        # Features
        batch_features = [item["features"] for item in batch]

        # Labels
        batch_classes = [item["classes"] for item in batch]
        batch_classes_all = [item["classes_all"] for item in batch]
        batch_classes_one_hot = [item["classes_one_hot"] for item in batch]

        batch_goal = [item["goal"] for item in batch]
        batch_goal_one_hot = [item["goal_one_hot"] for item in batch]
        
        # Masks
        batch_mask_past = [item["mask_past"] for item in batch]
        batch_mask_future = [item["mask_future"] for item in batch]

        # META INFO
        file_names = np.asarray([item["file_name"] for item in batch])
        batch_vid_len = [item["vid_len"] for item in batch]

        batch_video_files = [item["video_file"] for item in batch]

        """ PAD """
        # PADDING (based on the LONGEST sequence in the batch)
        bz = len(batch_features)
        length_of_seq = list(map(len, batch_classes))
        length_of_seq_all = list(map(len, batch_classes_all))

        features_tensor = torch.zeros(bz, np.shape(batch_features[0])[0], max(length_of_seq), dtype=torch.float)  # B x D x T_max
        
        
        goals_tensor = torch.ones(bz, 1, dtype=torch.long) * (self.num_highlevel_classes + 1)
        classes_tensor = torch.ones(bz, max(length_of_seq), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_all = torch.ones(bz, max(length_of_seq_all), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_one_hot = torch.zeros(bz, self.num_classes, max(length_of_seq), dtype=torch.float)
        goals_tensor_one_hot = torch.zeros(bz, self.num_highlevel_classes, max(length_of_seq), dtype=torch.float)

        mask_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_past_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_future_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)

        for i in range(bz):
            features_tensor[i, :, : np.shape(batch_features[i])[1]] = torch.from_numpy(batch_features[i])

            classes_tensor[i, : np.shape(batch_classes[i])[0]] = torch.from_numpy(batch_classes[i])
            classes_tensor_all[i, : np.shape(batch_classes_all[i])[0]] = torch.from_numpy(batch_classes_all[i])
            classes_tensor_one_hot[i, :, : np.shape(batch_classes_one_hot[i])[1]] = torch.from_numpy(batch_classes_one_hot[i])

            mask_tensor[i, 0, : np.shape(batch_classes[i])[0]] = torch.ones(np.shape(batch_classes[i])[0])
            mask_past_tensor[i, 0, : np.shape(batch_mask_past[i])[0]] = torch.from_numpy(batch_mask_past[i])
            mask_future_tensor[i, 0, : np.shape(batch_mask_future[i])[0]] = torch.from_numpy(batch_mask_future[i])

            goals_tensor[i, : np.shape(batch_goal[i])[0]] = torch.from_numpy(batch_goal[i])
            goals_tensor_one_hot[i, :, : np.shape(batch_goal_one_hot[i])[1]] = torch.from_numpy(batch_goal_one_hot[i])

        # SORT BY LENGTH and PERMUTE
        # lengths = torch.tensor(length_of_seq)
        vid_lengths = torch.tensor(batch_vid_len)
        _, perm_idx = torch.sort(torch.tensor(length_of_seq), 0, descending=True)
        vid_lengths = vid_lengths[perm_idx]

        features_tensor = features_tensor[perm_idx] #(8, 2048, 61)
        
        classes_tensor = classes_tensor[perm_idx]
        
        classes_tensor_all = classes_tensor_all[perm_idx]
        classes_tensor_one_hot = classes_tensor_one_hot[perm_idx]
        
        mask_tensor = mask_tensor[perm_idx]
        mask_past_tensor = mask_past_tensor[perm_idx]
        mask_future_tensor = mask_future_tensor[perm_idx]

        # META INFO
        file_names = file_names[perm_idx.tolist()]
        meta_dict = {"file_names": file_names}

        video_files = [batch_video_files[i] for i in perm_idx.tolist()]

        return (
            features_tensor,
            classes_tensor,
            classes_tensor_all,
            classes_tensor_one_hot,
            mask_tensor,
            mask_past_tensor,
            mask_future_tensor,
            vid_lengths,
            meta_dict,
            goals_tensor,
            goals_tensor_one_hot,
            video_files,
        )
