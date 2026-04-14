import os, json, pickle, random, argparse
import numpy as np

LABEL_MAP = {
    'normal':          0,
    'fighting':        1,
    'bullying_attack': 2,
    'bullying_victim': 2,
    'falling':         3,
    'climbing':        4,
    'vandalism':       5,
    'self_harm':       6,
}

CLIP_LEN = 48
STRIDE   = 16
MAX_PERSON = 2

def interpolate_low_conf(kps_xy, kps_score, threshold=0.3):
    """对置信度低的关键点做线性插值（逐关节处理）"""
    kps_xy = np.array(kps_xy)
    kps_score = np.array(kps_score)
    T, J = kps_score.shape
    for j in range(J):
        valid = np.where(kps_score[:, j] >= threshold)[0]
        if len(valid) == 0:
            continue
        invalid = np.where(kps_score[:, j] < threshold)[0]
        if len(invalid) == 0:
            continue
        kps_xy[invalid, j, 0] = np.interp(invalid, valid, kps_xy[valid, j, 0])
        kps_xy[invalid, j, 1] = np.interp(invalid, valid, kps_xy[valid, j, 1])
    return kps_xy, kps_score

def json_to_samples(json_path):
    """把一个视频的json切成多个48帧样本"""
    with open(json_path) as f:
        data = json.load(f)

    label_name = data['label']
    if label_name not in LABEL_MAP:
        print(f"  跳过未知类别: {label_name}")
        return []

    label_int = LABEL_MAP[label_name]
    img_shape = tuple(data['img_shape'])

    frames = data['frames']
    all_tids = set()
    for fdata in frames.values():
        all_tids.update(fdata.keys())
    all_tids = [int(t) for t in all_tids]

    if not all_tids:
        return []

    frame_indices = sorted(int(k) for k in frames.keys())
    total_frames = max(frame_indices) + 1

    tid_kps = {tid: {} for tid in all_tids}
    for fi_str, fdata in frames.items():
        fi = int(fi_str)
        for tid_str, pdata in fdata.items():
            tid = int(tid_str)
            tid_kps[tid][fi] = (np.array(pdata['kps']), np.array(pdata['score']))

    tid_counts = {tid: len(v) for tid, v in tid_kps.items()}
    top_tids = sorted(tid_counts, key=lambda t: -tid_counts[t])[:MAX_PERSON]

    samples = []
    video_name = os.path.splitext(os.path.basename(json_path))[0]

    for start in range(0, total_frames - CLIP_LEN + 1, STRIDE):
        end = start + CLIP_LEN
        clip_frames = list(range(start, end))

        # keypoint 只有 (x, y)，不包含 score
        keypoint = np.zeros((MAX_PERSON, CLIP_LEN, 17, 2), dtype=np.float32)
        keypoint_score = np.zeros((MAX_PERSON, CLIP_LEN, 17), dtype=np.float32)

        for p_idx, tid in enumerate(top_tids):
            kps_seq = []
            score_seq = []
            for fi in clip_frames:
                if fi in tid_kps[tid]:
                    kps_seq.append(tid_kps[tid][fi][0])
                    score_seq.append(tid_kps[tid][fi][1])
                else:
                    kps_seq.append(np.zeros((17, 2)))
                    score_seq.append(np.zeros(17))

            kps_arr, score_arr = interpolate_low_conf(kps_seq, score_seq)

            for t in range(CLIP_LEN):
                for j in range(17):
                    keypoint[p_idx, t, j, 0] = kps_arr[t, j, 0]
                    keypoint[p_idx, t, j, 1] = kps_arr[t, j, 1]
                    keypoint_score[p_idx, t, j] = score_arr[t, j]

        sample = {
            'frame_dir': f'{video_name}_clip_{start}',
            'label': label_int,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'total_frames': CLIP_LEN,
            'keypoint': keypoint,  # (2, 48, 17, 2)
            'keypoint_score': keypoint_score,  # (2, 48, 17)
        }
        samples.append(sample)

    return samples

def build_dataset(data_dir, out_dir, val_ratio=0.2):
    all_samples = []

    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        print(f"[{label_name}] {len(json_files)} 个视频")
        for jf in json_files:
            samples = json_to_samples(os.path.join(label_dir, jf))
            all_samples.extend(samples)
            print(f"  {jf}: {len(samples)} 个样本")

    random.shuffle(all_samples)
    split = int(len(all_samples) * (1 - val_ratio))
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(out_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_samples, f)

    print(f"\n完成：train={len(train_samples)} 样本，val={len(val_samples)} 样本")
    print(f"输出目录：{out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw_skeletons')
    parser.add_argument('--out_dir',  default='data/campus')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    args = parser.parse_args()
    build_dataset(args.data_dir, args.out_dir, args.val_ratio)