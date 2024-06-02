'''
@author: enpei
将SAHI核心逻辑封装成类，方便调用
对视频进行切分检测
1. 读取视频
2. 切分
3. batch detect
4. 对检测结果进行后处理，去除重复框，去除小框，去除低置信度框
5. 画框
6. 保存视频
'''

from ultralytics import YOLO
import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sahi.postprocess.utils import ObjectPrediction
from sahi.postprocess.combine import GreedyNMMPostprocess

# 命令行设置 export YOLO_VERBOSE=false 

class videoSAHI:
    def __init__(self):

        # 加载检测模型
        self.detection_model = YOLO("./weights/yolov8s.pt")  
        # 获取类别 
        self.objs_labels = self.detection_model.names 
        # 打印类别
        print(self.objs_labels)

        # detection threshold
        self.conf_thresh = 0.6
        # 随机颜色列表
        self.colors_list = np.random.randint(0, 255, size=(len(self.objs_labels), 3), dtype=np.uint8).tolist()

        self.postprocess = GreedyNMMPostprocess(
            match_threshold=0.7,
            match_metric="IOS",
            class_agnostic=False,
        )

  
    def getColorsList(self, num_colors):
        '''
        生成颜色列表
        '''
        hexs = ('FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB', '00C2FF',
                '2C99A8', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7', 'FF3838', 'FF9D97')
        # hex to bgr
        bgr_list = []
        for hex in hexs:
            bgr_list.append(tuple(int(hex[i:i+2], 16) for i in (4, 2, 0)))
        # 随机取num_colors个颜色
        # final_list = [random.choice(bgr_list) for i in range(num_colors)]  
        return bgr_list  
        
    def get_slice_bboxes(
        self,
        image_height: int,
        image_width: int,
        slice_height: Optional[int] = None,
        slice_width: Optional[int] = None,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
    ) -> List[List[int]]:
       
        slice_bboxes = []
        y_max = y_min = 0

        if slice_height and slice_width:
            y_overlap = int(overlap_height_ratio * slice_height)
            x_overlap = int(overlap_width_ratio * slice_width)
       
        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes
  
    def main(self):
        '''
        主函数
        '''
        # 读取视频
        cap = cv2.VideoCapture('media/4k.mp4')
  
        # 获取视频帧率、宽、高
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"fps: {fps}, width: {width}, height: {height}")
        # 保存视频
        writer = cv2.VideoWriter("output/4k_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # 切片大小
        slice_height = 1000
        slice_width = 1000
        # 切分，只需要做一次，返回的是位置信息
        slice_bboxes = self.get_slice_bboxes(height, width, slice_height= slice_height, slice_width=slice_width, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
        # 打印切分数量
        print(f"slice_bboxes: {len(slice_bboxes)}")

        # 批次大小，如果GPU显存不够，可以调小
        batch_size = 8

        while True:
            # 读取一帧
            start_time = time.time()
            ret, frame = cap.read()
            # 遍历每个切分框
            input_imgs = []
            # 记录每个切分框的左上角坐标
            slice_bbox_lt = []
            for slice_bbox in slice_bboxes:
                xmin, ymin, xmax, ymax = slice_bbox
                # 截取切分框
                slice_img = frame[ymin:ymax, xmin:xmax]
                input_imgs.append(slice_img)
                slice_bbox_lt.append([xmin, ymin])
                # save image
                # cv2.imwrite(f"output/{xmin}_{ymin}_{xmax}_{ymax}.jpg", slice_img)
            input_imgs.append(frame)
            # batch detect
            # 分批，每批batch_size张图片
            result_list = []
            for i in range(0, len(input_imgs), batch_size):
                input_imgs_batch = input_imgs[i:i+batch_size]
                # 参数说明：批量图片、是否流式、置信度阈值、iou阈值、是否使用半精度、最大检测框数、检测的类别（只检测person,bicycle,car,motorbike,bus,truck）
                results = self.detection_model(input_imgs_batch, stream=False, conf=self.conf_thresh, iou=0.7, half = False, max_det = 10, classes = [0,1,2,3,5,7])
                # 拼接结果
                result_list.extend(results)
            # print(f"result_list: {len(result_list)}")
            # 遍历结果
            all_boxes = []
            for i, result in enumerate(result_list):
                boxes = result.boxes  
                boxes = boxes.cpu().numpy() 
                # 遍历每个框
                
                for box in boxes.data:
                    l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
                    conf, class_id = box[4:] # confidence, class

                    if i < len(result_list) - 1: # 不是最后一张图片，即切片图片，需要转换为原图坐标
                        # 转换为原图坐标，平移
                        xmin, ymin = slice_bbox_lt[i]
                        l += xmin
                        t += ymin
                        r += xmin
                        b += ymin
                        # 绘制框
                        # cv2.rectangle(frame, (l, t), (r, b), self.colors_list[int(class_id)], 4)
                    else:
                        # 原始的图片，不需要转换坐标
                        # cv2.rectangle(frame, (l, t), (r, b), (0,0,255), 4)
                        pass
                    
                    # 添加到objectprediction
                    obj_item = ObjectPrediction(
                        bbox=[l, t, r, b],
                        score=conf,
                        category_id=int(class_id),
                    )
                    all_boxes.append(obj_item)
                    # put text
                    # text = f"{self.objs_labels[int(class_id)]} {conf:.2f}"
                    # cv2.putText(frame, text, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors_list[int(class_id)], 2)
            
            # 对所有的框进行后处理，去除重复框，去除小框，去除低置信度框
            all_boxes_processed = self.postprocess(all_boxes)

            # 遍历每个框
            for box in all_boxes_processed:
                l,t,r,b = box.bbox.to_xyxy()
                conf = box.score.value
                class_id = box.category.id

                # 绘制框
                cv2.rectangle(frame, (l, t), (r, b), self.colors_list[int(class_id)], 4)
                # put text
                text = f"{self.objs_labels[int(class_id)]} {conf:.2f}"
                cv2.putText(frame, text, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors_list[int(class_id)], 2)
            


            # fps
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            cv2.putText(frame, "FPS: %.2f" % fps, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
            # 写入视频
            writer.write(frame)
            # 显示，resize to 1/4
            frame = cv2.resize(frame, (width//4, height//4))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release
        cap.release()
        writer.release()


# main
if __name__ == '__main__':
    videoSAHI().main()