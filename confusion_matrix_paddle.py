# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:07:08 2021

@author: KenYH_Hsu
"""

import numpy as np
from os import listdir
from os.path import isfile, isdir, join


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    iou = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
    
    # KenHsu add
    for i in range(len(iou)):
        for j in range(len(iou[i])):
            if iou[i][j] > 0 and iou[i][j] < 0.5:
                if inter[i][j] / area2[j] > 0.5:
                    iou[i][j] = 0.5 + (iou[i][j] * 0.1)
    # KenHsu add
    
    return iou#inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.5, IOU_THRESHOLD = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''
        detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        gt_classes = labels[:, 0].astype(np.int16)
        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)
        
        for i in range(len(all_ious[0])) :
            if False not in (all_ious[:, i] < 0.1):
                self.matrix[3, 3] += 1

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
        
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0: # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index = True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index = True)[1]]


        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[(gt_class), detection_class] += 1 #TP
            else:
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, (gt_class)] += 1 #FN
        
        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1 #FP 
            elif all_matches.shape[0] == 0 and len(detection) > 0:
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1 #FP
        

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))
            
            
if __name__ == '__main__':
    result = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    detection = []
    gt = []
    gt_num = 0
    
    result_path = r"D:\PaddleDetection\output"
    # result_path = r"D:\PaddleDetection\relabel\training_1723\faster_rcnn_r101"
    # result_path = r"D:\PyTorch_YOLOv5\data\lesion4.names"
    # result_path = r"D:\for_testing\test"
    
    # gt_xml_path = r"D:\for_testing\NUMS\XML"
    gt_xml_path = r"E:\20211117\test\Annotations"
    # gt_xml_path = r"E:\ReLabel\VOCdevkit_test_20211220_original_just_2class\Annotations"
    # gt_xml_path = r"E:\gfm70_Data\gfm70-20211124\Annotations"
    
    
    files = sorted(listdir(result_path))
    for file in files:
        if file.find('.jpg') > 0:
            fullpath = join(result_path, file.replace('.jpg', '.txt'))
            try :
                f = open(fullpath, 'r')
                text = f.read()
                lines = text.split('\n')
                if '' in lines:
                    lines.remove('')
                
                for i in range(len(lines)):
                    line = lines[i].split(' ')
                    
                    line[0] = 0 if line[0] == 'focal' else 1
                    line = [float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[1]), line[0]]
                    
                    x = line[0]
                    y = line[1]
                    w = line[2]
                    h = line[3]
                    
                    line[2] = x + w
                    line[3] = y + h
                    
                    detection.append(line)
                    
                if detection == []:
                    detection = [[0, 0, 0, 0, 0, 0]]
            except :
                detection = [[0, 0, 0, 0, 0, 0]]
                
            
            fullpath = join(gt_xml_path, file.replace('.jpg', '.xml'))
            f = open(fullpath, 'r')
            text = f.read()
            lines = text.split('\n')
            for i in range(len(lines)):
                if lines[i].find('<object>') > 0:
                    xmin = lines[i+6].split("<xmin>")[1].split("</xmin>")[0]
                    ymin = lines[i+7].split("<ymin>")[1].split("</ymin>")[0]
                    xmax = lines[i+8].split("<xmax>")[1].split("</xmax>")[0]
                    ymax = lines[i+9].split("<ymax>")[1].split("</ymax>")[0]
                    Type = lines[i+1].split("<name>")[1].split("</name>")[0]
                    if Type == 'focal': Type = 0
                    elif Type == 'bifurcation': Type = 1
                    else: Type = 2
                    if float(xmax) - float(xmin) < 0:
                        temp = xmax
                        xmax = xmin
                        xmin = temp
                        print(fullpath)
                    if float(ymax) - float(ymin) < 0:
                        temp = ymax
                        ymax = ymin
                        ymin = temp
                        print(fullpath)
                    
                    gt.append([Type, xmin, ymin, xmax, ymax])
            try:
                # print(detection)
                # print('\n')
                # print(gt)
                gt_num = gt_num + len(gt)
                conf_mat = ConfusionMatrix(num_classes = 3, CONF_THRESHOLD = 0.5, IOU_THRESHOLD = 0.5)
                conf_mat.process_batch(np.array(detection).astype(np.float32), np.array(gt).astype(np.float32))
                result = result + conf_mat.return_matrix()
                # print(result)
            except Exception as e:
                print(e)
            gt = []
            detection = []
    
    focal_tp = int(result[0][0])
    focal_fp = int(result[0][3] + result[1][0])
    focal_fn = int(result[3][0] + result[0][1])
    focal_precision = focal_tp / (focal_tp + focal_fp)
    focal_recall = focal_tp / (focal_tp + focal_fn)
    print('conf: ' + str(conf_mat.CONF_THRESHOLD) + '  IOU: ' + str(conf_mat.IOU_THRESHOLD))
    print('       TP\tFN\tFP\t\tPercision\tRecall')
    print('focal: %s\t%s\t%s\t\t%3f\t%3f' %(focal_tp, focal_fn, focal_fp, focal_precision, focal_recall))
    
    bifu_tp = int(result[1][1])
    bifu_fp = int(result[1][3] + result[0][1])
    bifu_fn = int(result[3][1] + result[1][0])
    bifu_precision = bifu_tp / (bifu_tp + bifu_fp)
    bifu_recall = bifu_tp / (bifu_tp + bifu_fn)
    print('bifu:  %s\t%s\t%s\t\t%3f\t%3f' %(bifu_tp, bifu_fn, bifu_fp, bifu_precision, bifu_recall))
    
    all_tp = focal_tp + bifu_tp
    all_fp = focal_fp + bifu_fp
    all_fn = focal_fn + bifu_fn
    all_precision = all_tp / (all_tp + all_fp)
    all_recall = all_tp / (all_tp + all_fn)
    print('all:   %s\t%s\t%s\t\t%3f\t%3f' %(all_tp, all_fn, all_fp, all_precision, all_recall))
    print('FP類別預測錯誤: ' + str(int(focal_fp + bifu_fp - result[3][3])))
    print('FP位置預測錯誤: ' + str(int(result[3][3])))
    print('FN完全沒預測到: ' + str(int(focal_fn + bifu_fn - (focal_fp + bifu_fp - result[3][3]))))
    
    
    # gt_xml_path = r"D:\for_testing\ALL_XML"
    # # gt_xml_path = r"D:\for_testing\test"
    # files = sorted(listdir(gt_xml_path))
    # for file in files:
    #      fullpath = join(gt_xml_path, file)
    #      f = open(fullpath, 'r')
    #      text = f.read()
    #      lines = text.split('\n')
         
    #      for i in range(len(lines)):
    #          if lines[i].find('<object>') > 0:
    #              xmin = lines[i+6].split("<xmin>")[1].split("</xmin>")[0]
    #              ymin = lines[i+7].split("<ymin>")[1].split("</ymin>")[0]
    #              xmax = lines[i+8].split("<xmax>")[1].split("</xmax>")[0]
    #              ymax = lines[i+9].split("<ymax>")[1].split("</ymax>")[0]
    #              Type = lines[i+1].split("<name>")[1].split("</name>")[0]
    #              if Type == 'focal': Type = 0
    #              else: Type = 1
                 
    #              if float(xmax) - float(xmin) < 0:
    #                  temp = xmax
    #                  xmax = xmin
    #                  xmin = temp
    #                  print(fullpath)
    #              if float(ymax) - float(ymin) < 0:
    #                  temp = ymax
    #                  ymax = ymin
    #                  ymin = temp
    #                  print(fullpath)
                 
    #              gt.append([Type, xmin, ymin, xmax, ymax])
    #      break
     
    # conf_mat = ConfusionMatrix(num_classes = 2, CONF_THRESHOLD = 0.1, IOU_THRESHOLD = 0.5)
    # conf_mat.process_batch(np.array(detection).astype(np.float32), np.array(gt).astype(np.float32))
    # result = conf_mat.print_matrix()