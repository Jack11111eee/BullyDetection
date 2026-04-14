import pickle

# 加载原始训练集和验证集样本列表
with open('/home/hzcu/BullyDetection/data/campus/train.pkl', 'rb') as f:
    train_samples = pickle.load(f)
with open('/home/hzcu/BullyDetection/data/campus/val.pkl', 'rb') as f:
    val_samples = pickle.load(f)

# 合并所有样本到一个大的 annotations 列表中
all_samples = train_samples + val_samples

# 使用 frame_dir 作为唯一标识符存入 split
# 这样即便 annotations 顺序变化，也能通过 frame_dir 匹配到样本
train_ids = [s['frame_dir'] for s in train_samples]
val_ids   = [s['frame_dir'] for s in val_samples]

# 构造 PySKL 标准格式
data = {
    'split': {
        'train': train_ids, 
        'val': val_ids
    },
    'annotations': all_samples
}

# 保存合并后的 pkl 文件
output_file = '/home/hzcu/BullyDetection/data/campus/campus.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print(f'成功！训练集: {len(train_ids)} 条, 验证集: {len(val_ids)} 条')
print(f'文件已保存至: {output_file}')