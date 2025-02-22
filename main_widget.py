# -*- coding: utf-8 -*-
import sys
import os
import time
import torch
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from image_widget import ImageWidget
from common_utils import get_api_from_model, search, load_itk_image, truncate_hu, normalazation
import threading
import qdarkstyle
from glob import glob
import json
import pdb
import cv2
import numpy as np
import queue
from matplotlib import pyplot as plt

# ui配置文件
cUi, cBase = uic.loadUiType("v1_main.ui")

# 主界面
class MainWidget(QWidget, cUi):
    log_sig = pyqtSignal(str)
    
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        
        # read info
        with open('./info.json', 'r') as f:
            self.info = json.load(f)

        # init title
        self.setWindowTitle(self.info['version'])
        icon = QIcon()
        icon.addPixmap(QPixmap('./icons/logo.png'))
        self.setWindowIcon(icon)

        # init imagewidget
        self.cImageWidget = ImageWidget()
        self.cImageWidget.set_alg_handle(self)
        self.tabWidget.insertTab(0, self.cImageWidget, "Prediction Window")
        
        # slider
        # self.Slider.valueChanged.connect(self.changeImage)

        # init config widget
        self.btnSaveCfg.hide()

        # show imagewidget
        self.tabWidget.setCurrentIndex(0)
    
        # init treewidget
        self.treeModel.header().setVisible(False)
        
        # init log
        self.log_sig.connect(self.slot_log_info)
        
        self.alg = None
        self.alg_name = None
        self.model_name = None
        self.model_cfg = None
        self.model_cfg_widget = {}
        self.alg_model_map = {}
        self.det_thread_flag = True
        self.det_thread_queue = queue.Queue(maxsize=2)
        self.det_thread_handle = threading.Thread(target=self.det_thread_func, args=())
        self.det_thread_handle.start()
        self.update_model_flag = False
        self.create_model_process = 0
        self.create_process_dialog = None
        self.slice_list = []
        self.is_file = False
        self.hidesignal = None

    # @pyqtSlot()
    # def on_btnHide_clicked(self):
    #     self.hidesignal = 1 if self.hidesignal == 0 or self.hidesignal == None else 0

    @pyqtSlot()
    def on_btnPhoto_clicked(self):
        self.is_file = False
        self.slice_list = []
        print('on_btnPhoto_clicked')
        img_path = QFileDialog.getOpenFileName(self,  "Choose one file", "./inputs", "Images (*.jpg);;Images (*.png)") 
        img_path = img_path[0]
        if img_path != '':
            self.cImageWidget.slot_photo_frame(img_path)

    @pyqtSlot()
    def on_btnFile_clicked(self):
        print('on_btnFile_clicked')
        self.is_file = True
        self.slice_list = []
        img_path = QFileDialog.getOpenFileName(self,  "Choose one file", "./inputs", "Images (*.dcm);;Images (*.mhd)")
        img_path = img_path[0]
        

        if img_path.split(".")[-1] == 'dcm':
            img_path = "/".join(img_path.split("/")[:-1]) + "/*.dcm"
            self.Slider.setMaximum(len(glob(img_path))-1)
            if img_path != '':
                self.slice_list = self.cImageWidget.slot_file_frame(img_path)          
        
        elif img_path.split(".")[-1] == 'mhd':
            fileDir = []
            fileDir = search(path=r"D:/LUNA/seg-lungs-LUNA16", name = ".".join(img_path.split("/")[-1].split(".")[:-1]), fileDir=fileDir)
            for file in fileDir:
                if '.mhd' in file:
                    segmask_filedir = file
                    break
            numpyimage, _, _ = load_itk_image(img_path)
            numpymask, _, _ = load_itk_image(segmask_filedir)
            numpymask[np.where(numpymask!=0)]=1
            truncate_hu(numpyimage)
            numpyimage = normalazation(numpyimage) * 255

            self.Slider.setMaximum(numpyimage.shape[0]-1)
            if img_path != '':
                self.slice_list = [{"image": np.stack([numpyimage.transpose(1,2,0)[:,:,i]]*3, axis=2).astype(np.uint8),
                                    "mask": np.stack([numpymask.transpose(1,2,0)[:,:,i]]*3, axis=2).astype(np.uint8)} 
                                    for i in range(numpyimage.shape[0])]


    def det_thread_func(self):
        self.log_sig.emit('Checking...')
        
        # search all algs
        self.search_alg_and_model()
        
        lastslider = -1
        while self.det_thread_flag:
            if self.update_model_flag:
                self.updaet_model()
                self.update_model_flag = False

            if self.is_file == True and len(self.slice_list) > 0:
                if lastslider == self.Slider.value():
                    continue
                if self.model_name == 'nodule':
                    self.cImageWidget.cAlg.add_img(self.slice_list[self.Slider.value()]['image'])
                else:
                    self.cImageWidget.cAlg.add_img(self.slice_list[self.Slider.value()])

                lastslider = self.Slider.value()

            try:
                img = self.det_thread_queue.get(block=True, timeout=0.2)
            except queue.Empty:
                img = None

            if img is not None and self.alg is not None:    
                start_time = time.time()

                if self.model_name == 'nodule':
                    ret = self.alg.inference(img, segmask=self.slice_list[self.Slider.value()]['mask'])
                else:
                    ret = self.alg.inference(img)


                if self.cImageWidget is not None:
                    time_spend = time.time()-start_time
                    if 'result' not in self.model_cfg.keys():
                        save_result = 0
                        save_path = None
                    else:
                        save_result = int(self.model_cfg['result']['save_result'])
                        save_path = self.model_cfg['result']['save_dir']
                    self.cImageWidget.slot_alg_result(img, ret, time_spend, save_result, save_path)

    def slot_log_info(self, info):
        if str(info).startswith('cmd:'):
            if 'load models finished' in str(info):
                self.init_model_tree()
            if 'start create model' in str(info):
                self.tabWidget.setCurrentIndex(0)
                self.cImageWidget.change_background('start_load')               
            if 'create model failed' in str(info):
                self.cImageWidget.change_background('load_fail')
            if 'create model success' in str(info):
                self.cImageWidget.change_background('load_success')
            if 'pretrain unget' in str(info):
                box_message = str(info).split('=')[-1]
                box = QMessageBox()
                box.setIcon(QMessageBox.Critical)
                box.setTextInteractionFlags(Qt.TextSelectableByMouse)
                box.setWindowTitle(u"Modol does not exist")
                box.setText(box_message)
                box.setTextInteractionFlags(Qt.TextSelectableByMouse)
                box.exec()
            if 'update title' in str(info):
                title_name = str(info).split('=')[-1]
                self.setWindowTitle(title_name)
        elif str(info).startswith('news_id'):
            self.tabWidget.setTabIcon(2, QIcon(QPixmap("./icons/news.png")))
        else:
            self.logEdit.append('<font color="#FF9090">%s</font>'%(info))    
                
    def check_news(self, x):
        lines = x.split('\n')
        for line in lines:
            if 'news_id' in line:
                id = int(line.split(':')[-1])
                if id != self.info['news_id']:
                    self.info['news_id'] = id
                    self.has_news = True
                    with open('./info.json', 'w') as f:
                        json.dump(self.info, f)
                    self.log_sig.emit('news_id')
                    break
                
        
    def add_img(self, img):
        if self.det_thread_queue.full():
            return
        else:
            self.det_thread_queue.put(img)
        
    def search_alg_and_model(self):
        self.alg_model_map = {}
        self.log_sig.emit('Initialization, please wait...')
        for sub_dir in os.listdir('./model_zoo'):
            self.log_sig.emit('>> Loading model: %s'%str(sub_dir))
            sub_path = os.path.join('./model_zoo', sub_dir)
            if os.path.isdir(sub_path):
                api = get_api_from_model(str(sub_dir))
                if api is not None:
                    self.alg = api.Alg()
                    self.alg_model_map[str(sub_dir)] = self.alg.get_support_models()
                    self.log_sig.emit('>> Loading model: %s success'%str(sub_dir))
                else:
                    self.alg_model_map[str(sub_dir)] = []
                    self.log_sig.emit('>> Loading model: %s failed'%str(sub_dir))
        self.log_sig.emit('Load models finished\n')
        self.log_sig.emit('cmd:load models finished')

    def init_model_tree(self):
        for alg in self.alg_model_map.keys():
            item_alg = QTreeWidgetItem(self.treeModel)
            #item_alg.setFlags(Qt.ItemIsEnabled)
            item_alg.setText(0, alg)
            for model in self.alg_model_map[alg]:
                item_model = QTreeWidgetItem(item_alg)
                item_model.setText(0, model)
                    
    def updaet_model(self):
        self.log_sig.emit('cmd:start create model')
        self.log_sig.emit('Start to create model: %s'%str(self.model_name))
        self.log_sig.emit('>> Stop ImageWidget')
        self.cImageWidget.stop_all()
        title_name = self.info['version'] + '(Current model: ' + self.model_name + ";"

        ##
        # self.alg_name = "YoloV5"; self.model_cfg['normal']['weight'] = "yolo5x.pt"

        pretrain_path = './model_zoo/' + self.alg_name + '/' + self.model_cfg['normal']['weight']
        if not os.path.exists(pretrain_path):
            self.log_sig.emit('  Create model: %s failed，pretrained weight not download'%str(self.model_name))
            box_info = u'Download weight and put it under model_zoo/%s \nAddress:\n%s'%(self.alg_name, self.model_cfg['normal']['url'])
            self.log_sig.emit('cmd:pretrain unget=%s'%box_info)
            self.alg = None
            return
        if self.alg is not None:
            device = 'cuda' if self.model_cfg['device']['dev_type'] == 'gpu' else 'cpu'
            title_name += ' device: ' + device + ")"
            self.log_sig.emit('>> device:' + device)
            self.alg.create_model(self.model_name, device)
            self.log_sig.emit('cmd:create model success')
            self.log_sig.emit('  Create model: %s finish\n'%str(self.model_name))
        else:
            self.log_sig.emit('cmd:create model failed')
            self.log_sig.emit('  Create model: %s failed，algorithm not create'%str(self.model_name))
            self.alg = None
        self.log_sig.emit('cmd:update title=%s'%(title_name))

    def _translate_str(self, ori_str):
        translate_map = {'device': 'Device',
                         'dev_type': 'cpu/gpu',
                         'result': 'Result',
                         'save_result': 'If save result',
                         'save_dir': 'Save dir',
                         'normal': 'Parameters',
                         }
        if ori_str in translate_map.keys():
            return translate_map[ori_str]
        else:
            return ori_str

    def _init_cfg_widget(self):
        old_items = []
        for i in range(self.cfg_layout.count()):
            old_items.append(self.cfg_layout.itemAt(i))
            
        for old_item in old_items:
            self.cfg_layout.removeItem(old_item) 
 
        self.model_cfg_widget = {}
        if self.alg is not None:
            self.btnSaveCfg.show()
            self.model_cfg = self.alg.get_model_cfg(self.model_name)
            for key in self.model_cfg.keys():
                label_title = QLabel()
                label_title.setText('<font color="#FF9090">%s</font>'%(self._translate_str(key)))
                self.cfg_layout.addWidget(label_title)
                self.model_cfg_widget[key] = {}
                for sub_key in self.model_cfg[key]:
                    frame = QFrame()
                    edit_layout = QHBoxLayout()
                    edit_key = QLineEdit()
                    edit_value = QLineEdit()
                    edit_key.setText(self._translate_str(sub_key))
                    edit_key.setReadOnly(False)
                    edit_key.setFocusPolicy(Qt.NoFocus)
                    edit_value.setText(str(self.model_cfg[key][sub_key]))
                    edit_layout.addWidget(edit_key)
                    edit_layout.addWidget(edit_value)
                    edit_layout.setStretch(0, 1)
                    edit_layout.setStretch(1, 2)
                    self.cfg_layout.addLayout(edit_layout)
                    self.model_cfg_widget[key][sub_key] = edit_value
                label_space = QLabel()
                self.cfg_layout.addWidget(label_space)
            
            spacer = QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.cfg_layout.addItem(spacer)
            
    def _get_cfg_widget(self):
        for key in self.model_cfg_widget.keys():
            for sub_key in self.model_cfg_widget[key].keys():
                edit_widget = self.model_cfg_widget[key][sub_key]
                old_cfg_value = self.model_cfg[key][sub_key]
                new_cfg_value = edit_widget.text()
                if sub_key == 'dev_type':
                    if new_cfg_value != 'cpu':
                        if not torch.cuda.is_available():
                            reply = QMessageBox.warning(self,
                                u'警告', 
                                u'当前pytorch不支持cuda, 将创建cpu模型', 
                                QMessageBox.Yes)
                            edit_widget.setText('cpu')
                            new_cfg_value = 'cpu'
                self.model_cfg[key][sub_key] = new_cfg_value

    def on_treeModel_itemClicked(self, item, seq):
        print(item.text(0), item.parent())
        if item.parent() is None:
            print('you select alg')
        else:
            print('yolo select model: ', item.parent().text(0), item.text(0))
            self.alg_name = item.parent().text(0)
            self.model_name = item.text(0)
            api = get_api_from_model(self.alg_name)
            if api is None:
                self.alg = None
                print('error, the api can not import')
            else:
                self.alg = api.Alg()
                self._init_cfg_widget()
                #self.updaet_model()
                self.update_model_flag = True
    
    @pyqtSlot()
    def on_btnSaveCfg_clicked(self):
        print('button btnSaveCfg clicked')
        self._get_cfg_widget()
        self.alg.put_model_cfg(self.model_name, self.model_cfg)
        #self.updaet_model()
        self.update_model_flag = True

    def closeEvent(self, event):        
        # reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
        #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # if reply == QMessageBox.Yes:
        event.accept()
        self.cImageWidget.stop_all()
        self.det_thread_flag = False
        self.det_thread_handle.join()
        # else:
        #     event.ignore()

        
if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cMainWidget = MainWidget()
    cApp.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    cMainWidget.show()
    sys.exit(cApp.exec_())                                                                                
