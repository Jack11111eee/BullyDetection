"""
数据平衡脚本 v2：
- 大类欠采样到 CAP
- 小类过采样但不超过 MAX_OVERSAMPLE 倍
- 去掉 self_harm
"""
import pickle
import random
import copy
from collections import Counter

random.seed(42)

ann_file = '/home/hzcu/BullyDetection/data/campus/campus.pkl'
out_file = '/home/hzcu/BullyDetection/data/campus/campus_balanced.pkl'

CLASSES = ['normal', 'fighting', 'bullying', 'falling', 'climbing', 'vandalism']
CAP = 6000            # 大类最多保留
MAX_OVERSAMPLE = 3    # 小类最多复制几倍

with open(ann_file, 'rb') as f:
    data = pickle.load(f)

train_ids = set(data['split']['train'])
val_ids = set(data['split']['val'])

train_by_class = {}
val_anns = []

for ann in data['annotations']:
    if ann['frame_dir'] in train_ids:
        label = ann['label']
        train_by_class.setdefault(label, []).append(ann)
    elif ann['frame_dir'] in val_ids:
        val_anns.append(ann)

# 去掉 self_harm (label>=6)
train_by_class = {k: v for k, v in train_by_class.items() if k < 6}
val_anns = [a for a in val_anns if a['label'] < 6]

print("=" * 50)
print("平衡前（训练集）：")
for label in sorted(train_by_class):
    print(f"  {label} {CLASSES[label]:<12} {len(train_by_class[label]):>6}")

balanced_train = []
for label in sorted(train_by_class):
    samples = train_by_class[label]
    n = len(samples)

    if n >= CAP:
        # 欠采样
        balanced_train.extend(random.sample(samples, CAP))
    elif n * MAX_OVERSAMPLE <= CAP:
        # 小类：复制到 MAX_OVERSAMPLE 倍（不硬凑到 CAP）
        target = n * MAX_OVERSAMPLE
        balanced_train.extend(samples)
        extra = target - n
        for i in range(extra):
            s = copy.deepcopy(random.choice(samples))
            s['frame_dir'] = s['frame_dir'] + f'_dup{i}'
            balanced_train.append(s)
    else:
        # 中等类：复制到 CAP
        balanced_train.extend(samples)
        extra = CAP - n
        for i in range(extra):
            s = copy.deepcopy(random.choice(samples))
            s['frame_dir'] = s['frame_dir'] + f'_dup{i}'
            balanced_train.append(s)

random.shuffle(balanced_train)

print("\n" + "=" * 50)
print("平衡后（训练集）：")
counts = Counter(a['label'] for a in balanced_train)
for label in sorted(counts):
    orig = len(train_by_class[label])
    now = counts[label]
    ratio = now / orig
    print(f"  {label} {CLASSES[label]:<12} {orig:>6} -> {now:>6}  (x{ratio:.1f})")

print(f"\n训练集: {len(balanced_train)}, 验证集: {len(val_anns)}")

# 保存
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