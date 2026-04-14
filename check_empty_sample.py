import pickle
import numpy as np

pkl_file = '/home/hzcu/BullyDetection/data/campus/campus_balanced_v2.pkl'

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

empty_count = 0
single_person = 0
multi_person = 0

for ann in data['annotations']:
    keypoint = ann['keypoint']
    # 看有多少人
    try:
        if keypoint.ndim == 3:  # (T, K, 3)
            # 单人
            if np.all(np.isnan(keypoint)):
                empty_count += 1
            else:
                single_person += 1
        elif keypoint.ndim == 4:  # (M, T, K, 3) — 多人
            # 检查每个人
            for m in range(keypoint.shape[0]):
                if np.all(np.isnan(keypoint[m])):
                    empty_count += 1
                else:
                    multi_person += 1
    except:
        pass

print(f"空样本（全NaN）: {empty_count}")
print(f"单人样本: {single_person}")
print(f"多人样本: {multi_person}")