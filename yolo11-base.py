import argparse, json, os                                                                                             
import numpy as np
import cv2                                                                                                            
from ultralytics import YOLO                              


def get_img_shape(source):
      """获取视频或帧文件夹的分辨率"""
      if os.path.isdir(source):
          # 帧文件夹：读第一张图
          frames = sorted([f for f in os.listdir(source) if f.endswith(('.jpg', '.png'))])
          if not frames:
              return 1080, 1920
          img = cv2.imread(os.path.join(source, frames[0]))
          return img.shape[0], img.shape[1]
      else:
          cap = cv2.VideoCapture(source)
          h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          cap.release()
          return h, w


def extract_skeletons(video_path, label, out_path):
      model = YOLO('yolo11m-pose.pt')

      img_h, img_w = get_img_shape(video_path)

      # YOLO 的 track() 对视频文件和帧文件夹都支持，source 直接传即可
      results = model.track(
          source=video_path,
          persist=True,
          tracker='bytetrack.yaml',
          conf=0.3,
          iou=0.5,
          stream=True,
          verbose=False
      )

      frame_data = {}
      for frame_idx, result in enumerate(results):
          if result.boxes is None or result.keypoints is None:
              continue
          frame_data[frame_idx] = {}
          for i, box in enumerate(result.boxes):                                                                        
              if box.id is None:
                  continue                                                                                              
              tid = int(box.id.item())                      
              kps_raw = result.keypoints.data[i].cpu().numpy()  # (17, 3)
              frame_data[frame_idx][tid] = {
                  'kps':   kps_raw[:, :2].tolist(),
                  'score': kps_raw[:, 2].tolist()
              }

      out = {
          'label':     label,
          'img_shape': [img_h, img_w],                                                                                  
          'frames':    frame_data
      }                                                                                                                 
      with open(out_path, 'w') as f:                        
          json.dump(out, f)
      print(f"  保存 {len(frame_data)} 帧 → {out_path}")


if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--video',  required=True, help='视频文件路径 或 帧文件夹路径')
      parser.add_argument('--label',  required=True)
      parser.add_argument('--out',    required=True)
      args = parser.parse_args()
      extract_skeletons(args.video, args.label, args.out)