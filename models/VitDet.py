import numpy as np
import os, cv2,sys
import torch, detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.config import LazyConfig,instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
setup_logger()


def load_model(model_path): #mask_rcnn_vitdet_h_100ep.py
  cfg = LazyConfig.load("/content/detectron2/projects/ViTDet/configs/LVIS/mask_rcnn_vitdet_h_100ep.py")
  model =  instantiate(cfg.model)
  DetectionCheckpointer(model).load(model_path)
  return model

def Draw_pred_BB(img,model_result):
  final_img = img.copy()
  for row_index in range(len(model_result[0]["instances"])):
    if model_result[0]["instances"].scores[row_index] > 0.4:
      x1,y1,x2,y2 = model_result[0]["instances"].pred_boxes[row_index].tensor.int().tolist()[0]
      cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
  
def Draw_original_seg(img,model_result):
  v = Visualizer(img, scale=1.2)
  out = v.draw_instance_predictions(model_result[0]["instances"].to("cpu"))
  cv2.imshow("bounding_box", out.get_image()[:, :, ::-1])
  cv2.waitKey(0)
  cv2.destroyAllWindows()