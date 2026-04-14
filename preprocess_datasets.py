# 统一处理所有数据集，自动调用 step3

import os
import subprocess
import numpy as np
import hashlib

                                                    
RLVS_DIR     = '/home/hzcu/zjc/dataset/NTU/RLVS/Real Life Violence Dataset'                            
RWF_DIR      = '/home/hzcu/zjc/dataset/RWF-2000/RWF-2000'
SHT_DIR      = '/home/hzcu/zjc/dataset/SHT/SHT/shanghaitech'
UCF_DIR      = '/home/hzcu/zjc/dataset/UCF-crime'
URFALL_DIR = '/home/hzcu/zjc/dataset/Fall UR Fall Detection'
CHUTE_DIR     = '/home/hzcu/zjc/dataset/Fall Multiple Cameras Fall Dataset/dataset_extracted/dataset'                 
FALLFLOOR_DIR = '/home/hzcu/zjc/dataset/fall_floor/fall_floor_extracted/fall_floor'                                   
VANDALISM2_DIR= '/home/hzcu/zjc/dataset/Vandalism/Vandalism_extracted/Vandalism'                                      
MULTICAM_DIR  = '/home/hzcu/zjc/dataset/fall Multiple Cameras Fall Dataset/downloads'                                 
PUNCH_DIR     = '/home/hzcu/zjc/dataset/punch/punch_extracted/punch'
CLIMB_DIR = '/home/hzcu/zjc/dataset/climb/climb_extracted/climb'

OUT_DIR      = '/home/hzcu/BullyDetection/data/raw_skeletons'
STEP3_SCRIPT = 'yolo11-base.py'                                                                                     
  
  # UCF-Crime 类别映射（只保留我们需要的）                                                                              
UCF_LABEL_MAP = {                                         
      'Fighting':        'fighting',
      'Assault':         ['bullying_attack', 'bullying_victim'],  # 跑两遍
      'Vandalism':       'vandalism',
  }

  # SHT testing 异常视频对应标签（SHT 的异常主要是打架/追逐）
SHT_ABNORMAL_LABEL = 'fighting'
  # ============================================================


def safe_stem(name, prefix='', max_len=60):
    """如果文件名过长，则将其替换为 MD5 哈希值"""
    stem = name
    candidate = f"{prefix}{stem}.json"
    # 如果编码为 utf-8 后的字节长度超过 200，则进行压缩
    if len(candidate.encode('utf-8')) > 200:
        h = hashlib.md5(name.encode('utf-8')).hexdigest()[:12]
        stem = h
    return f"{prefix}{stem}"

def run_step3(source, label, out_path):
      """调用 step3"""
      os.makedirs(os.path.dirname(out_path), exist_ok=True)
      if os.path.exists(out_path):
          print(f"  已存在，跳过: {out_path}")                                                                          
          return
      cmd = [                                                                                                           
          'python', STEP3_SCRIPT,                           
          '--video', source,
          '--label', label,
          '--out', out_path
      ]
      print(f"  处理: {source} → [{label}]")
      subprocess.run(cmd, check=True)                                                                                   
  
                                                                                                                        
  # ============================================================
  # RLVS
  # ============================================================
def process_rlvs():
      print("\n=== RLVS ===")
      for split in ['Violence', 'NonViolence']:
          label = 'fighting' if split == 'Violence' else 'normal'
          split_dir = os.path.join(RLVS_DIR, split)
          if not os.path.isdir(split_dir):
              print(f"  找不到目录: {split_dir}")
              continue
          for fname in sorted(os.listdir(split_dir)):
              if not fname.endswith('.mp4'):
                  continue                                                                                              
              video_path = os.path.join(split_dir, fname)
              out_path = os.path.join(OUT_DIR, label, safe_stem(fname.replace(".mp4",""), prefix='rlvs_') + '.json')                         
              run_step3(video_path, label, out_path)        


  # ============================================================
  # RWF-2000
  # ============================================================
def process_rwf():
      print("\n=== RWF-2000 ===")
      for split in ['train', 'val']:
          for category in ['Fight', 'NonFight']:
              label = 'fighting' if category == 'Fight' else 'normal'
              cat_dir = os.path.join(RWF_DIR, split, category)
              if not os.path.isdir(cat_dir):
                  print(f"  找不到目录: {cat_dir}")
                  continue
              for fname in sorted(os.listdir(cat_dir)):
                  if not fname.endswith('.avi'):
                      continue
                  video_path = os.path.join(cat_dir, fname)
                  stem = fname.replace('.avi', '')
                  out_path = os.path.join(OUT_DIR, label, safe_stem(stem, prefix=f'rwf_{split}_') + '.json')
                  run_step3(video_path, label, out_path)


  # ============================================================
  # SHT
  # ============================================================
def get_sht_abnormal_frame_ranges(sht_dir):
      """
      读取 test_frame_mask 下的 .npy 文件，                                                                             
      返回 {folder_name: [异常帧索引列表]}
      异常帧 = mask 值为 1 的帧                                                                                         
      """                                                   
      mask_dir = os.path.join(sht_dir, 'testing', 'test_frame_mask')
      result = {}
      if not os.path.isdir(mask_dir):
          return result
      for fname in os.listdir(mask_dir):
          if not fname.endswith('.npy'):
              continue
          mask = np.load(os.path.join(mask_dir, fname))
          abnormal_frames = np.where(mask == 1)[0].tolist()                                                             
          key = fname.replace('.npy', '')
          result[key] = abnormal_frames                                                                                 
      return result                                         


def process_sht():
      print("\n=== SHT ===")

      # training 下全是正常行为视频                                                                                     
      train_dir = os.path.join(SHT_DIR, 'training','videos')
      if os.path.isdir(train_dir):                                                                                      
          for fname in sorted(os.listdir(train_dir)):       
              if not fname.endswith('.avi'):
                  continue
              video_path = os.path.join(train_dir, fname)
              stem = fname.replace('.avi', '')
              out_path = os.path.join(OUT_DIR, 'normal', safe_stem(stem ,prefix = f'sht_train_{stem}') + '.json')
              run_step3(video_path, 'normal', out_path)

      # testing 下的帧序列，用 frame_mask 区分正常/异常帧
      frames_dir = os.path.join(SHT_DIR, 'testing', 'frames')
      abnormal_map = get_sht_abnormal_frame_ranges(SHT_DIR)

      if not os.path.isdir(frames_dir):
          print("  找不到 SHT testing/frames 目录")
          return

      for folder in sorted(os.listdir(frames_dir)):
          folder_path = os.path.join(frames_dir, folder)
          if not os.path.isdir(folder_path):
              continue

          has_abnormal = folder in abnormal_map and len(abnormal_map[folder]) > 0
          label = SHT_ABNORMAL_LABEL if has_abnormal else 'normal'
          out_path = os.path.join(OUT_DIR, label, safe_stem(stem, prefix = f'sht_test_{folder}') + '.json')                                            
          run_step3(folder_path, label, out_path)
                                                                                                                        
                                                            
  # ============================================================
  # UCF-Crime
  # ============================================================
def process_ucf():                                                                                                    
      print("\n=== UCF-Crime ===")                                                                                      
      import shutil                                         
                                                                                                                        
      for split in ['Train', 'Test']:                       
          split_dir = os.path.join(UCF_DIR, split)                                                                      
          if not os.path.isdir(split_dir):                  
              print(f"  找不到目录: {split_dir}")                                                                       
              continue
                                                                                                                        
          for category in sorted(os.listdir(split_dir)):                                                                
              if category not in UCF_LABEL_MAP:
                  print(f"  跳过: {category}")                                                                          
                  continue                                  
                                                                                                                        
              cat_dir = os.path.join(split_dir, category)                                                               
              labels = UCF_LABEL_MAP[category]
              if isinstance(labels, str):                                                                               
                  labels = [labels]                                                                                     
   
              # 按视频名分组帧：Fighting002_x264_1000.png → video=Fighting002, frame=1000                               
              video_frames = {}                             
              for fname in os.listdir(cat_dir):
                  if not fname.endswith('.png') or '_x264_' not in fname:                                               
                      continue                                                                                          
                  parts = fname.split('_x264_')                                                                         
                  video_name = parts[0]                                                                                 
                  frame_num = int(parts[1].replace('.png', ''))
                  if video_name not in video_frames:                                                                    
                      video_frames[video_name] = []
                  video_frames[video_name].append((frame_num, os.path.join(cat_dir, fname)))                            
                                                            
              print(f"  [{split}/{category}] 找到 {len(video_frames)} 个视频")                                          
                                                            
              for video_name, frame_list in sorted(video_frames.items()):                                               
                  frame_list.sort(key=lambda x: x[0])  # 按帧号排序
                                                                                                                        
                  for label in labels:                      
                      out_path = os.path.join(OUT_DIR, label,
                                              safe_stem(f'ucf_{split.lower()}_{video_name}') + '.json')                 
                      if os.path.exists(out_path):
                          print(f"  已存在，跳过: {video_name} → [{label}]")                                            
                          continue                                                                                      
                                                                                                                        
                      # 建临时文件夹，帧用软链接按顺序放进去                                                            
                      tmp_dir = f'/tmp/ucf_{video_name}_{label}'
                      os.makedirs(tmp_dir, exist_ok=True)                                                               
                      for idx, (_, fpath) in enumerate(frame_list):                                                     
                          link = os.path.join(tmp_dir, f'{idx:06d}.png')                                                
                          if not os.path.exists(link):                                                                  
                              os.symlink(fpath, link)                                                                   
                                                                                                                        
                      run_step3(tmp_dir, label, out_path)                                                               
                      shutil.rmtree(tmp_dir)  # 用完删掉

# ============================================================
  # UR Fall Detection Dataset
  # ============================================================
def process_urfall():
      print("\n=== UR Fall Detection ===")
      label_map = {
          'Fall':   'falling',
          'Normal': 'normal',
      }
      for folder, label in label_map.items():
          folder_path = os.path.join(URFALL_DIR, folder)
          if not os.path.isdir(folder_path):
              print(f"  找不到目录: {folder_path}")
              continue
          for fname in sorted(os.listdir(folder_path)):
              if not fname.endswith('.mp4'):
                  continue
              video_path = os.path.join(folder_path, fname)
              stem = fname.replace('.mp4', '')
              out_path = os.path.join(OUT_DIR, label, safe_stem(stem, prefix=f'urfall_{folder.lower()}_') + '.json')
              if os.path.exists(out_path):
                  print(f"  已存在，跳过: {out_path}")
                  continue

              # 裁剪右半边RGB，保存为临时文件
              tmp_path = f'/tmp/urfall_rgb_{stem}.mp4'
              crop_rgb_half(video_path, tmp_path)
              run_step3(tmp_path, label, out_path)
              os.remove(tmp_path)  # 用完删掉临时文件


def crop_rgb_half(src_path, dst_path):
      """把左右拼接视频裁出右半边（RGB部分）"""
      import cv2
      cap = cv2.VideoCapture(src_path)
      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)
      half_w = w // 2

      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      out = cv2.VideoWriter(dst_path, fourcc, fps, (half_w, h))

      while True:
          ret, frame = cap.read()
          if not ret:
              break
          out.write(frame[:, half_w:, :])  # 取右半边

      cap.release()
      out.release()

# ============================================================                                                        
  # Chute Fall Dataset（Fall Multiple Cameras Fall Dataset）
  # ============================================================
def process_chute():
      print("\n=== Chute Fall Dataset ===")                                                                             
      if not os.path.isdir(CHUTE_DIR):
          print(f"  找不到目录: {CHUTE_DIR}")                                                                           
          return                                                                                                        
      for chute in sorted(os.listdir(CHUTE_DIR)):
          chute_path = os.path.join(CHUTE_DIR, chute)                                                                   
          if not os.path.isdir(chute_path) or not chute.startswith('chute'):
              continue                                                                                                  
          for fname in sorted(os.listdir(chute_path)):
              if not fname.endswith('.avi'):                                                                            
                  continue                                                                                              
              video_path = os.path.join(chute_path, fname)
              stem = f'{chute}_{fname.replace(".avi", "")}'                                                             
              out_path = os.path.join(OUT_DIR, 'falling', safe_stem(stem, prefix='chute_') + '.json')                   
              run_step3(video_path, 'falling', out_path)                                                                
                                                                                                                        
                                                                                                                        
  # ============================================================                                                        
  # Fall Floor Dataset                      
  # ============================================================
def process_fallfloor():
      print("\n=== Fall Floor Dataset ===")
      if not os.path.isdir(FALLFLOOR_DIR):                                                                              
          print(f"  找不到目录: {FALLFLOOR_DIR}")
          return                                                                                                        
      for fname in sorted(os.listdir(FALLFLOOR_DIR)):
          if not fname.endswith('.avi'):  # .avi.tform.mat 不以 .avi 结尾，自动过滤                                     
              continue                                                                                                  
          video_path = os.path.join(FALLFLOOR_DIR, fname)                                                               
          stem = fname.replace('.avi', '')                                                                              
          out_path = os.path.join(OUT_DIR, 'falling', safe_stem(stem, prefix='fallfloor_') + '.json')
          run_step3(video_path, 'falling', out_path)                                                                    
                                            
                                                                                                                        
  # ============================================================
  # Vandalism Dataset（新增）
  # ============================================================
def process_vandalism2():
      print("\n=== Vandalism Dataset ===")                                                                              
      if not os.path.isdir(VANDALISM2_DIR):
          print(f"  找不到目录: {VANDALISM2_DIR}")                                                                      
          return                            
      for fname in sorted(os.listdir(VANDALISM2_DIR)):
          if not fname.endswith('.mp4'):                                                                                
              continue
          video_path = os.path.join(VANDALISM2_DIR, fname)                                                              
          stem = fname.replace('.mp4', '').replace('_x264', '')
          out_path = os.path.join(OUT_DIR, 'vandalism', safe_stem(stem, prefix='vandalism2_') + '.json')                
          run_step3(video_path, 'vandalism', out_path)                                                                  
                                                                                                                        
                                                                                                                        
  # ============================================================
  # fall Multiple Cameras Fall Dataset（Subject/Activity/Trial/时间戳帧）
  # ============================================================                                                        
def process_multicam_fall():
      print("\n=== fall Multiple Cameras Fall Dataset ===")                                                             
      import shutil                         
      if not os.path.isdir(MULTICAM_DIR):                                                                               
          print(f"  找不到目录: {MULTICAM_DIR}")
          return                                                                                                        
      for subject in sorted(os.listdir(MULTICAM_DIR)):
          subject_path = os.path.join(MULTICAM_DIR, subject)
          if not os.path.isdir(subject_path):                                                                           
              continue
          for activity in sorted(os.listdir(subject_path)):                                                             
              activity_path = os.path.join(subject_path, activity)
              if not os.path.isdir(activity_path):                                                                      
                  continue
              for trial in sorted(os.listdir(activity_path)):                                                           
                  trial_path = os.path.join(activity_path, trial)
                  if not os.path.isdir(trial_path):
                      continue                                                                                          
  
                  frames = sorted([f for f in os.listdir(trial_path) if f.endswith('.png')])                            
                  if not frames:            
                      continue                                                                                          
  
                  stem = f'{subject}_{activity}_{trial}'                                                                
                  out_path = os.path.join(OUT_DIR, 'falling', safe_stem(stem, prefix='multicam_') + '.json')
                  if os.path.exists(out_path):                                                                          
                      print(f"  已存在，跳过: {stem}")
                      continue                                                                                          
                                            
                  tmp_dir = f'/tmp/multicam_{subject}_{activity}_{trial}'                                               
                  os.makedirs(tmp_dir, exist_ok=True)
                  for idx, fname in enumerate(frames):                                                                  
                      link = os.path.join(tmp_dir, f'{idx:06d}.png')
                      if not os.path.exists(link):                                                                      
                          os.symlink(os.path.join(trial_path, fname), link)                                             
  
                  run_step3(tmp_dir, 'falling', out_path)                                                               
                  shutil.rmtree(tmp_dir)    
                                                                                                                        
                                            
  # ============================================================
  # Punch Dataset → bullying_attack
  # ============================================================
def process_punch():                                                                                                  
      print("\n=== Punch Dataset ===")      
      if not os.path.isdir(PUNCH_DIR):
          print(f"  找不到目录: {PUNCH_DIR}")
          return                                                                                                        
      for fname in sorted(os.listdir(PUNCH_DIR)):
          if not fname.endswith('.avi'):                                                                                
              continue                      
          video_path = os.path.join(PUNCH_DIR, fname)
          stem = fname.replace('.avi', '')                                                                              
          for label in ['bullying_attack', 'bullying_victim']:
              out_path = os.path.join(OUT_DIR, label, safe_stem(stem, prefix='punch_') + '.json')                       
              run_step3(video_path, label, out_path)

# ============================================================
  # Climb Dataset                                                                                                       
  # ============================================================                                                        
def process_climb():                                          
      print("\n=== Climb Dataset ===")                                                                                  
      if not os.path.isdir(CLIMB_DIR):      
          print(f"  找不到目录: {CLIMB_DIR}")
          return                             
      for fname in sorted(os.listdir(CLIMB_DIR)):                                                                       
          if not fname.endswith('.avi'):  # .mat 自动过滤
              continue                                                                                                  
          video_path = os.path.join(CLIMB_DIR, fname)
          stem = fname.replace('.avi', '')           
          out_path = os.path.join(OUT_DIR, 'climbing', safe_stem(stem, prefix='climb_') + '.json')                      
          run_step3(video_path, 'climbing', out_path)

  # ============================================================
  # 主入口
  # ============================================================
if __name__ == '__main__':
      process_rlvs()
      process_rwf()
      process_sht()
      process_ucf()
      process_urfall()
      
      process_chute()                 
      process_fallfloor()                                                                                         
      process_vandalism2()   
      process_multicam_fall()                                                                                     
      process_punch() 
      process_climb()      

      print("\n全部完成，可以跑 build_pkl.py 了。")