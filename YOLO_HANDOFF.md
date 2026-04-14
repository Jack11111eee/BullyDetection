# YOLO11s 小物体检测 — 对接文档

> 给队友的接口规范，确保训练完的模型能直接对接主系统。

---

## 一、任务说明

训练 YOLO11s 检测校园监控中的小物体，用于规则引擎判断吸烟、打电话等行为。

---

## 二、需要检测的目标类别

| 类别ID | 类别名 | 用途 |
|--------|--------|------|
| 0 | cigarette | 吸烟检测 |
| 1 | phone | 打电话检测 |

> 如果数据集中有其他有用类别（如 lighter、knife），可以一起加进来，但上面两个是必须的。

---

## 三、推荐数据来源

- Roboflow Universe 搜索 `cigarette detection`（YOLO 格式导出）
- Roboflow Universe 搜索 `cell phone detection`（YOLO 格式导出）
- 导出格式选 **YOLOv8 / YOLO11**（txt 标注）

---

## 四、训练要求

```bash
# 基础模型
yolo detect train model=yolo11s.pt data=your_dataset.yaml \
  imgsz=1280 epochs=100 batch=16 device=0
```

关键参数：
- **imgsz=1280**（必须，俯视角监控中香烟/手机很小，640 检测不到）
- **model=yolo11s.pt**（s 版本够用，速度快）
- **conf 阈值**：推理时建议 0.3，后续规则引擎会再过滤

---

## 五、交付物

训练完后需要提供：

### 1. 模型权重（必须）
```
best.pt  — 训练后的最优权重文件
```
放到 `/home/hzcu/BullyDetection/yolo11s_smallobj.pt`

### 2. 类别映射（必须）
告知类别顺序，例如：
```python
# 模型输出的 class_id 对应关系
SMALL_OBJ_CLASSES = {
    0: 'cigarette',
    1: 'phone',
    # 如果有更多类别继续加
}
```

### 3. 推荐的推理参数（建议提供）
```python
conf = 0.3    # 置信度阈值
iou = 0.5     # NMS 的 IoU 阈值
imgsz = 1280  # 推理分辨率
```

---

## 六、主系统调用方式

我这边会这样调用你的模型：

```python
from ultralytics import YOLO

# 加载模型
small_obj_model = YOLO('yolo11s_smallobj.pt')

# 对每一帧检测
results = small_obj_model(frame, conf=0.3, iou=0.5, imgsz=1280, verbose=False)

# 解析结果
for box in results[0].boxes:
    cls_id = int(box.cls.item())      # 类别ID
    conf = float(box.conf.item())     # 置信度
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 框坐标
```

返回结果会进入规则引擎，和 YOLO11m-Pose 的骨骼关键点配合判断：
- **吸烟**：检测到 cigarette + 在人手/嘴附近
- **打电话**：检测到 phone + 在人耳朵附近

---

## 七、时间线

- PoseC3D 行为识别模型正在训练中（约 12 小时）
- 你的 YOLO11s 训练完后，直接把 `best.pt` 放到上面的路径
- 我这边写好规则引擎后会做端到端联调

有问题随时沟通。
