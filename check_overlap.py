"""check_overlap.py"""
import pickle

with open('data/campus/campus.pkl', 'rb') as f:
    data = pickle.load(f)

train_ids = set(data['split']['train'])
val_ids = set(data['split']['val'])
overlap = train_ids & val_ids

print(f"train: {len(train_ids)}, val: {len(val_ids)}")
print(f"重叠 frame_dir: {len(overlap)}")

if overlap:
    # 看重叠样本的标签分布
    from collections import Counter
    overlap_labels = [a['label'] for a in data['annotations'] if a['frame_dir'] in overlap]
    print(f"重叠样本标签分布: {Counter(overlap_labels)}")