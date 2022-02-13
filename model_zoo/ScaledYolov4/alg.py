import sys
import os
import time
from pathlib import Path
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import numpy as np
from common_utils import vis
from common_utils import AlgBase
import yaml
import pdb

class Alg(AlgBase):
    def __init__(self):
        self.cfg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg.yaml")
        self.cfg_info = {}
        self.model_name = None
        self.model = None
        self.device = "cpu"
        self.ignore_keys = []
        self.load_cfg()
        
    def create_model(self, model_name="pancreas", dev="cpu"):
        if model_name not in self.cfg_info.keys():
            print('unknown model name:', model_name, 'create failed')
            return
        self.device = dev
        self.model_name = model_name

         # Load model
        pth_name = self.cfg_info[model_name]['normal']['weight']
        pre_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s'%(pth_name))
        if not os.path.exists(pre_train):
            return 'error: weight is not download, please download it from: %s'%self.cfg_info[model_name]['normal']['url']

        self.model = attempt_load(pre_train, map_location='cpu')  # load FP32 model
        self.model.eval()

        if self.device == 'cuda':
            self.model.cuda()

        return None
    
    def inference(self, img_array, segmask=None):
        map_result = {'type':'img'}
        if segmask is not None:
            img_mask = img_array * segmask
        else:
            img_mask = img_array
        img_resize = cv2.resize(img_mask,  tuple(self.cfg_info[self.model_name]['normal']['infer_size']))
        img = (img_resize - self.cfg_info[self.model_name]['normal']['mean']) / self.cfg_info[self.model_name]['normal']['std']
        img = img.transpose((2,0,1))    
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

        if self.device == "cuda":
            img_tensor = img_tensor.cuda()
        with torch.no_grad():
            pred = self.model(img_tensor, augment=False)[0]
            pred = non_max_suppression(pred, 
                                       float(self.cfg_info[self.model_name]['normal']['infer_conf']), 
                                       float(self.cfg_info[self.model_name]['normal']['nms_thre']), 
                                       classes=None, 
                                       agnostic=False)
        if pred == [None]:
            pred = [torch.zeros(0, 6)]
        valid_pred = pred[0].cpu()
        boxes = valid_pred[:,0:4]
        cls = valid_pred[:, 5]
        scores = valid_pred[:, 4]
        x_rate = img_mask.shape[1] /  self.cfg_info[self.model_name]['normal']['infer_size'][0]
        y_rate = img_mask.shape[0] /  self.cfg_info[self.model_name]['normal']['infer_size'][1]
        
        # no need for this line
        if self.model_name == 'pancreas':
            img_array = np.stack([np.flipud(img_array[...,0])]*3, axis=2)

        boxes[:,0:4:2] = boxes[:,0:4:2] * x_rate
        
        # no need for 512 - 
        if self.model_name == 'pancreas':
            boxes[:,1:4:2] = 512 - boxes[:,1:4:2] * y_rate
        else:
            boxes[:,1:4:2] = boxes[:,1:4:2] * y_rate


        vis(img_array, boxes, scores, cls, conf=float(self.cfg_info[self.model_name]['normal']['infer_conf']), class_names=self.cfg_info[self.model_name]['normal']['class_names'], model_name=self.model_name)
        map_result['result'] = img_array
        return map_result
