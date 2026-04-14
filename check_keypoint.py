import pickle
with open('/home/hzcu/BullyDetection/data/campus/campus_balanced_v2.pkl', 'rb') as f:
    data = pickle.load(f)
ann = data['annotations'][0]
print(f"keypoint shape: {ann['keypoint'].shape}")
print(f"keypoint dtype: {ann['keypoint'].dtype}")
print(f"keypoint sample:\n{ann['keypoint'][:5, :5]}")  # 前5帧、前5个关键点