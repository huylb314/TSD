from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
import time
import json

import numpy as np

import glob
import os

os.system("rm -rf /result/")

config_file = 'faster_rcnn_r152_fpn_TSD_1x_zalo.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'save_models/faster_rcnn_r152_fpn_TSD_1x.pth'
public_test = "/data/*.png"
save_folder = "faster_rcnn_r152_fpn_TSD_1x_zalo_latest/"
save_submission = "/result/"
score_thr = 0.3

mmcv.mkdir_or_exist(save_folder)
mmcv.mkdir_or_exist(save_submission)

class_names = ["1. No entry", "2. No parking / waiting", \
               "3. No turning", "4. Max Speed", \
               "5. Other prohibition signs", "6. Warning", "7. Mandatory"]

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

submit_list = []

for idx, test_file_path in enumerate(glob.glob(public_test)):
    print (idx)
    img_file = os.path.basename(test_file_path)
    img_name, _ = img_file.split('.')
    print (img_name)
    
    img_im_temp = mmcv.imread(test_file_path)
    result = inference_detector(model, test_file_path)
    img_out, out_bboxes, out_labels = show_result(test_file_path, result, class_names, show=False)
    out_scores = out_bboxes[:, -1]
    inds = out_scores > score_thr
    out_bboxes = out_bboxes[inds, :]
    out_labels = out_labels[inds]
    
    mmcv.imwrite(img_out, os.path.join(save_folder, img_file))
    
    for idx, (bbox, label) in enumerate(zip(out_bboxes, out_labels)):
        temp_ = dict()
        temp_["image_id"] = int(img_name)
        bbox_elem0, bbox_elem1, bbox_elem2, bbox_elem3, score = bbox
        bbox_width = bbox_elem2 - bbox_elem0
        bbox_height = bbox_elem3 - bbox_elem1
        
        temp_["category_id"] = int(label) + 1
        temp_["bbox"] = [float(item) for item in [round(bbox_elem0, 2), round(bbox_elem1, 2), round(bbox_width, 2), round(bbox_height, 2)]]
        temp_["score"] = float(score)
        print (bbox_elem0, bbox_elem1, bbox_elem2, bbox_elem3)
        print ((round(bbox_elem0, 2)), round(bbox_elem1, 2), round(bbox_width, 2), round(bbox_height, 2))
        print ('{:.2f} {:.2f} {:.2f} {:.2f}'.format(bbox_elem0, bbox_elem1, bbox_width, bbox_height))
        print (label)

        submit_list.append(temp_)

with open(os.path.join(save_submission, "submission.json"), "w")  as out_file:
    json.dump(submit_list, out_file, indent = 6)