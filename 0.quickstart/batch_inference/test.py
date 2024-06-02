# %%

# 导入所需的库
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import Video


# %%
# 加载模型：yolov8s coco预训练模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="weights/yolov8s.pt",
    confidence_threshold=0.5,
    device="cuda:0", 
)

# %%
# 打开测试视频
Video("media/4k.mp4", width=400)

# %%
# label_id对应的类别名称
category_names = list(detection_model.category_names)

# %%
# 打印前10个类别
category_names[:10]

# %%
# 使用分割预测
result = get_sliced_prediction(
    "media/snapshot.png",
    detection_model,
    slice_height = 1000,
    slice_width = 1000,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
    verbose = 2
)
