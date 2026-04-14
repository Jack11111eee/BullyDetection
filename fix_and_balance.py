import pickle
import random
import copy
from collections import Counter, defaultdict
import numpy as np

random.seed(42)

ann_file = '/home/hzcu/BullyDetection/data/campus/campus.pkl'
out_file = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v7.pkl'

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing']
CAP = 6000
MAX_OVERSAMPLE = 3

def get_video_name(frame_dir):                                                                                         
      """去掉 _clip_XXX"""                        
      idx = frame_dir.rfind('_clip_')                                                                                    
      return frame_dir[:idx] if idx >= 0 else frame_dir

def get_base_video(frame_dir):
      """去掉 _clip_XXX，再去掉 RWF 的人物索引 _N"""
      video = get_video_name(frame_dir)                                                                                  
      if video.startswith('rwf_'):                                                                                       
          # rwf_train_videoID_0 → rwf_train_videoID                                                                      
          parts = video.rsplit('_', 1)                                                                                   
          if parts[-1].isdigit():                                                                                        
              video = parts[0]
      return video

# ========== 读取原始数据 ==========
with open(ann_file, 'rb') as f:
    data = pickle.load(f)

# ========== 数据清洗 ==========
raw_anns = [a for a in data['annotations'] if a['label'] < 5]
print(f"过滤 label>=5 后: {len(raw_anns)}")

# 1. 过滤全零 keypoint
valid_anns = [a for a in raw_anns if not np.all(a['keypoint'] == 0)]
n_zero = len(raw_anns) - len(valid_anns)
print(f"过滤全零 keypoint: 移除 {n_zero} ({n_zero/len(raw_anns):.1%}), 剩余 {len(valid_anns)}")

# 2. 去重 frame_dir（同一 frame_dir 保留第一个）
seen = {}
for ann in valid_anns:
    fd = ann['frame_dir']
    if fd not in seen:
        seen[fd] = ann
all_anns = list(seen.values())
n_dup = len(valid_anns) - len(all_anns)
print(f"去重 frame_dir: 移除 {n_dup}, 最终有效样本 {len(all_anns)}")

# 清洗后类别分布
print("\n清洗后类别分布:")
for label in range(5):
    cnt = sum(1 for a in all_anns if a['label'] == label)
    print(f"  {label} {CLASSES[label]:<12} {cnt:>6}")
print(f"  总计: {len(all_anns)}")

# ========== 按 基础视频 分组，不按 label 分 ==========
base_to_anns = defaultdict(list)
for ann in all_anns:
    base = get_base_video(ann['frame_dir'])
    base_to_anns[base].append(ann)

all_bases = list(base_to_anns.keys())
random.shuffle(all_bases)
split_idx = int(len(all_bases) * 0.8)

train_bases = set(all_bases[:split_idx])
val_bases = set(all_bases[split_idx:])

train_anns = []
val_anns = []
for base, anns in base_to_anns.items():
    if base in train_bases:
        train_anns.extend(anns)
    else:
        val_anns.extend(anns)

# 验证无重叠
train_bases_check = set(get_base_video(a['frame_dir']) for a in train_anns)
val_bases_check = set(get_base_video(a['frame_dir']) for a in val_anns)
overlap = train_bases_check & val_bases_check
print(f"\n基础视频级别重叠: {len(overlap)}（应为 0）")
assert len(overlap) == 0

# 打印分布
print("\n划分结果：")
for label in range(5):
    t = sum(1 for a in train_anns if a['label'] == label)
    v = sum(1 for a in val_anns if a['label'] == label)
    print(f"  {label} {CLASSES[label]:<12}  train: {t:>6}  val: {v:>5}")

# ========== 平衡训练集 ==========
train_by_class = defaultdict(list)
for ann in train_anns:
    train_by_class[ann['label']].append(ann)

print("\n" + "=" * 60)
print("训练集平衡：\n")

balanced_train = []
for label in sorted(train_by_class):
    samples = train_by_class[label]
    n = len(samples)

    if n >= CAP:
        selected = random.sample(samples, CAP)
        balanced_train.extend(selected)
        tag = f"欠采样 {n} -> {CAP}"
    elif n * MAX_OVERSAMPLE <= CAP:
        target = n * MAX_OVERSAMPLE
        balanced_train.extend(samples)
        extra = target - n
        for i in range(extra):
            s = copy.deepcopy(random.choice(samples))
            s['frame_dir'] = s['frame_dir'] + f'_dup{i}'
            balanced_train.append(s)
        tag = f"过采样 {n} -> {target} (x{MAX_OVERSAMPLE})"
    else:
        balanced_train.extend(samples)
        extra = CAP - n
        for i in range(extra):
            s = copy.deepcopy(random.choice(samples))
            s['frame_dir'] = s['frame_dir'] + f'_dup{i}'
            balanced_train.append(s)
        tag = f"过采样 {n} -> {CAP}"

    print(f"  {label} {CLASSES[label]:<12}  {tag}")

random.shuffle(balanced_train)

# ========== 验证集分布 ==========
print(f"\n验证集（不做平衡）：")
val_counts = Counter(a['label'] for a in val_anns)
for label in sorted(val_counts):
    print(f"  {label} {CLASSES[label]:<12} {val_counts[label]:>6}")

print(f"\n最终: 训练集 {len(balanced_train)}, 验证集 {len(val_anns)}")

# ========== 保存 ==========
out_data = {
    'split': {
        'train': [s['frame_dir'] for s in balanced_train],
        'val':   [s['frame_dir'] for s in val_anns],
    },
    'annotations': balanced_train + val_anns,
}

with open(out_file, 'wb') as f:
    pickle.dump(out_data, f)

print(f"已保存: {out_file}")