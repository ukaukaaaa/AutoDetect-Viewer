import sys
import os
import torch
import importlib
import numpy as np
import cv2
from collections import OrderedDict
from ruamel import yaml
import SimpleITK as sitk
import time

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def del_all_model_zoo_modules():
    alg_names = []
    for dir in os.listdir('./model_zoo'):
        if os.path.isdir(os.path.join('./model_zoo',dir)):
            alg_names.append(str(dir))

    # del path in sys.path
    del_p = []
    for p in sys.path:
        for alg_name in alg_names:
            if alg_name in p:
                del_p.append(p)
                break
    for p in del_p:
        sys.path.remove(p)

    # del modeules
    old_alg_names = []
    all_keys = sys.modules.keys()
    time.sleep(0.2)
    for alg_name in all_keys:
        if 'from' in str(sys.modules[alg_name]) and \
            'model_zoo' in str(sys.modules[alg_name]) and \
            'YoloAll' in str(sys.modules[alg_name]):
            old_alg_names.append(alg_name)
        if 'namespace' in str(sys.modules[alg_name]) and hasattr(sys.modules[alg_name], '__path__'):
            module_path = str(sys.modules[alg_name].__path__)
            if 'model_zoo' in module_path and \
               'YoloAll' in module_path:
               old_alg_names.append(alg_name)
            
    for alg_name in old_alg_names:
        del sys.modules[alg_name] 

def add_one_model_path(alg_name):
    sub_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_zoo/'+ alg_name)
    #sys.path.append(sub_dir)
    sys.path.insert(0, sub_dir)
   
def get_api_from_model(alg_name):
    api = None
    del_all_model_zoo_modules()
    add_one_model_path(alg_name)
    
    try:
        api = importlib.import_module('model_zoo.%s.alg'%alg_name)
        print('create api from', alg_name, 'success')
    except ImportError as e:
        print('create api from', alg_name, 'failed')
        print('error:', str(e))   
        api = None
        
    return api

def load_pre_train_ignore_name(net, pre_train):
    if pre_train == '':
        print('the pre_train is null, skip')
        return
    else:
        print('the pre_train is %s' % pre_train)
        new_dict = {}
        pretrained_model = torch.load(pre_train, map_location=torch.device('cpu'))

        pre_keys = pretrained_model.keys()
        net_keys = net.state_dict().keys()
        print('net keys len:%d, pretrain keys len:%d' % (len(net_keys), len(pre_keys)))
        if len(net_keys) != len(pre_keys):
            print('key lens not same, maybe the pytorch version for pretrain and net are difficent; use name load')
            for key_net in net_keys:
                strip_key_net = key_net.replace('module.', '')
                if strip_key_net not in pre_keys:
                    print('op: %s not exist in pretrain, ignore' % (key_net))
                    new_dict[key_net] = net.state_dict()[key_net]
                    continue
                else:
                    net_shape = str(net.state_dict()[key_net].shape).replace('torch.Size', '')
                    pre_shape = str(pretrained_model[strip_key_net].shape).replace('torch.Size', '')
                    if net.state_dict()[key_net].shape != pretrained_model[strip_key_net].shape:
                        print('op: %s exist in pretrain but shape difficenet(%s:%s), ignore' % (
                        key_net, net_shape, pre_shape))
                        new_dict[key_net] = net.state_dict()[key_net]
                    else:
                        print(
                            'op: %s exist in pretrain and shape same(%s:%s), load' % (key_net, net_shape, pre_shape))
                        new_dict[key_net] = pretrained_model[strip_key_net]

        else:
            for key_pre, key_net in zip(pretrained_model.keys(), net.state_dict().keys()):
                if net.state_dict()[key_net].shape == pretrained_model[key_pre].shape:
                    new_dict[key_net] = pretrained_model[key_pre]
                    print('op: %s shape same, load weights' % (key_net))
                else:
                    new_dict[key_net] = net.state_dict()[key_net]
                    print('op: %s:%s shape diffient(%s:%s), ignore weights' %
                                    (key_net, key_pre,
                                    str(net.state_dict()[key_net].shape).replace('torch.Size', ''),
                                    str(pretrained_model[key_pre].shape).replace('torch.Size', '')))

        net.load_state_dict(new_dict, strict=False)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=COCO_CLASSES, model_name=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue

        
        """     
        True
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        
        """
        if model_name == 'pancreas':
            x0 = int(box[0])
            y1 = int(box[1])
            x1 = int(box[2])
            y0 = int(box[3])
        else:
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])            

        color = (COCO_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(COCO_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (COCO_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0-1, y0 - int(txt_size[1])),
            (x0 + txt_size[0], y0 - 1),
            # (x0, y0 + 1),
            # (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0-1, y0-2), font, 0.6, txt_color, thickness=2)

    return img

class AlgBase:
    def __init__(self):
        self.cfg_file = None
        self.ignore_keys = []

    def load_cfg(self):
        print(self.cfg_file)
        with open(self.cfg_file, "r", encoding='utf-8') as f:
            self.cfg_info  = yaml.round_trip_load(f)
        
    def save_cfg(self):
        with open(self.cfg_file, "w", encoding='utf-8') as f:
            yaml.round_trip_dump(self.cfg_info, f, default_flow_style=False)

    def get_model_cfg(self, model_name):
        cfg_map = {}
        for key in self.cfg_info[model_name].keys():
            cfg_map[key] = {}
            for sub_key in self.cfg_info[model_name][key].keys():
                if sub_key not in self.ignore_keys:
                    cfg_map[key][sub_key] = self.cfg_info[model_name][key][sub_key]
        return cfg_map
        
    def put_model_cfg(self, model_name, cfg_map):
        # the cfg_map is from qt widget , so they are all strings
        # i need to write them to cfg.yaml by my self
        #for key in cfg_map.keys():
        #    self.cfg_info[model_name][key] = cfg_map[key]
        old_lines = None
        with open(self.cfg_file, "r", encoding='utf-8') as f:
            old_lines = f.readlines()
        
        new_lines = []
        into_model_flag = False
        out_model_flag = False
        key_back = None
        for line in  old_lines:
            top_key = False if line.startswith(' ') else True
            key_name = line.split(':')[0].strip('\r\n').replace(' ', '')
            
            if key_name == model_name and into_model_flag is False:
                into_model_flag = True
                new_lines.append(line)
                continue
            elif into_model_flag and top_key:
                out_model_flag = True
                new_lines.append(line)
                continue
            
            '''
            if into_model_flag and not out_model_flag and key_name in cfg_map.keys():
                new_line = line.split(':')[0] + ': ' + cfg_map[key_name]
                if '\r' in line:
                    new_line = new_line + '\r'
                if '\n' in line:
                    new_line = new_line + '\n'
                new_lines.append(new_line)
            else:
                new_lines.append(line)
            '''

            if into_model_flag and not out_model_flag:
                if line.startswith(' ') and not line.startswith('   '):
                    key_back = key_name
                    new_lines.append(line)
                elif line.startswith('   '):
                    if key_back in cfg_map.keys() and key_name in cfg_map[key_back].keys():
                        new_line = line.split(':')[0] + ': ' + cfg_map[key_back][key_name]
                        if '\r' in line:
                            new_line = new_line + '\r'
                        if '\n' in line:
                            new_line = new_line + '\n'
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    assert(False)
            else:
                new_lines.append(line)

        with open(self.cfg_file, "w", encoding='utf-8') as f:
            f.writelines(new_lines)

        self.load_cfg()

    def get_support_models(self):
        model_list=[]
        for key in self.cfg_info.keys():
            if key != 'alg_info':
                model_list.append(key)
        return model_list

# save the info of CT
class CTImage(object):
    """docstring for Hotel"""
    def __init__(self, x_offset, y_offset, z_offset, x_ElementSpacing, y_ElementSpacing, z_ElementSpacing):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset
        self.x_ElementSpacing = x_ElementSpacing
        self.y_ElementSpacing = y_ElementSpacing
        self.z_ElementSpacing = z_ElementSpacing
        self.ElementSpacing = np.array([x_ElementSpacing, y_ElementSpacing, z_ElementSpacing])


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        offset = [k for k in contents if k.startswith('Offset')][0]
        EleSpacing = [k for k in contents if k.startswith('ElementSpacing')][0]
        
        offArr = np.array(offset.split(' = ')[1].split(' ')).astype('float')
        eleArr = np.array(EleSpacing.split(' = ')[1].split(' ')).astype('float')
        CT = CTImage(offArr[0],offArr[1],offArr[2],eleArr[0],eleArr[1],eleArr[2])
        transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transform = np.round(transform)
        if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False
    itkimage = sitk.ReadImage(filename)
    numpyimage = sitk.GetArrayFromImage(itkimage)
    if(isflip == True):
        numpyimage = numpyimage[:,::-1,::-1]
    return (numpyimage,CT,isflip)
 
# search mhd file in the folder
def search(path=".", name="", fileDir = []):
    
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search(item_path, name)
        elif os.path.isfile(item_path):            
            if name in item:
                fileDir.append(item_path)
                
    return fileDir

def truncate_hu(image_array):
    image_array[image_array > 400] = 400
    # image_array[image_array > 3072] = 3072
    
    image_array[image_array <-1000] = -1000
    # image_array[image_array <-1024] = -1024
    
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array - min)/(max - min)
    # image_array = (image_array - min)/(max - min)* 255
    # image_array = np.round(image_array)
    image_array = image_array
    return image_array 