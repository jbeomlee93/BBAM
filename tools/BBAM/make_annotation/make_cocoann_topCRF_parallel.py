import numpy as np
import os
import json
import cv2
import torch
from pycococreatortools import create_image_info, create_annotation_info
from pycocotools import mask
from tools.BBAM.make_annotation.anno_utils import get_final_mask_with_topCRF, get_same_n_proposal
import joblib
import time

# time.sleep(60*60*1)
VOC2012_JSON_FOLDER="Dataset/VOC2012_SEG_AUG/"
train_json = json.load(open(os.path.join(VOC2012_JSON_FOLDER, 'pascal_val2012.json')))

img_dir = 'Dataset/VOC2012_SEG_AUG/JPEGImages/'
img_list = open('Dataset/VOC2012_SEG_AUG/ImageSets/Main/train_aug_cocostyle.txt').readlines()
mask_name = 'BBAM_training_images'
mask_dir = mask_name

img_list = [i.strip() for i in img_list if os.path.exists(os.path.join(mask_dir, i.strip()))]

img_list = get_same_n_proposal(img_list)
th_fgs = [0.85, 0.2]


def process(img_name, th_fg):
    img_id = img_name.replace('_', '')
    img = cv2.imread(os.path.join(img_dir, img_name + '.jpg'))
    h, w = img.shape[:2]
    image_info = create_image_info(int(img_id), img_name + '.jpg', (w, h))
    ann = get_final_mask_with_topCRF(img_name, mask_dir, img, th_fg)

    return ann, image_info, w, h


for th_fg in th_fgs:
    coco_output = {}
    coco_output["images"] = []
    coco_output["annotations"] = []
    coco_output['categories'] = train_json['categories']

    coco_output['type'] = train_json['type']
    img_idx = 0
    instance_id = 1
    n_jobs = 80
    img_chunks = [img_list[i:i + n_jobs] for i in range(0, len(img_list), n_jobs)]
    start_time = time.time()
    for chunk_idx, img_chunk in enumerate(img_chunks):
        print("%s/%s doing..." % (chunk_idx, len(img_chunks)))
        results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(process)(i, th_fg) for i in img_chunk]
        )
        for ann_idx, ann in enumerate(results):
            ann, image_info, w, h = ann
            img_name = img_chunk[ann_idx]
            img_id = img_name.replace('_', '')
            coco_output["images"].append(image_info)

            if len(ann['box']) == 0:
                print("no box", img_name)
            else:
                for score, mask, class_id, box in zip(ann['score'], ann['mask'], ann['class'], ann['box']):
                    category_info = {'id': int(class_id), 'is_crowd': False}
                    box = [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1]

                    annotation_info_fg = create_annotation_info(
                        instance_id, int(img_id), category_info, mask, [w, h], tolerance=1,
                        bounding_box=torch.Tensor(box))

                    if annotation_info_fg == None:
                        continue

                    instance_id += 1
                    coco_output['annotations'].append(annotation_info_fg)
    print("time", time.time() - start_time)

    with open('datasets/%s_nimg_%d_th_%s.json' % (mask_name, len(img_list), th_fg), 'w') as outfile:
        json.dump(coco_output, outfile)
    print('datasets/%s_nimg_%d_th_%s.json' % (mask_name, len(img_list), th_fg))
# BBAM_training_images_nimg_10582_th_0.8.json