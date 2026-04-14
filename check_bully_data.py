import pickle
from collections import Counter

for path in ['data/campus/campus.pkl', 'data/campus/campus_balanced.pkl']:
      with open(path, 'rb') as f:
          d = pickle.load(f)
      val_ids = set(d['split']['val'])
      labels = [a['label'] for a in d['annotations'] if a['frame_dir'] in val_ids]
      print(f"\n{path}")
      for label, count in sorted(Counter(labels).items()):
          print(f"  label {label}: {count}")