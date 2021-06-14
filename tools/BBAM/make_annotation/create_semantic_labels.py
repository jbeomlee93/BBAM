import os
import cv2
import json
from pycocotools import coco
import numpy as np
from PIL import Image


train_json = 'BBAM_training_images_nimg_10582_th_0.85.json'
train_json_ignore = 'BBAM_training_images_nimg_10582_th_0.2.json'

img_root = 'Dataset/VOC2012_SEG_AUG/JPEGImages'
save_dir = 'Dataset/VOC2012_SEG_AUG/Semantic_labels'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
coco_class = coco.COCO(annotation_file=train_json)
coco_class_ignore = coco.COCO(annotation_file=train_json_ignore)

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]


total_img_ids = [[i['id'], i['file_name']] for i in coco_class.dataset["images"]]

def check_entire_overlap(target_box, boxes):
    box1 = target_box
    entire_overlap = False
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]

    for box2 in boxes:
        box2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]
        if not (box1 == box2):
            if (box2[0] <= box1[0]) and (box2[1] <= box1[1]) and (box2[2] >= box1[2]) and (box2[3] >= box1[3]):
                entire_overlap = True

    return entire_overlap


def box_area_sort(boxes):
    areas = [(box[2])*(box[3]) for box in boxes]
    return np.argsort(-np.array(areas))

def get_aggregated_mask(classes, boxes, masks):
    total_mask = np.zeros([21, masks[-1].shape[0], masks[-1].shape[1]])
    box_priority = box_area_sort(boxes)

    for box_idx in box_priority:
        cls, box, mask = classes[box_idx], boxes[box_idx], masks[box_idx]
        total_mask[cls, mask==1] = 1
    cls_mask = np.zeros(masks[-1].shape)
    for c in range(1, 21):
        cls_mask[total_mask[c]==1] = c
    cls_mask[np.sum(total_mask, axis=0)>1] = 255
    for box_idx in box_priority:
        if check_entire_overlap(boxes[box_idx], boxes):
            cls_mask[masks[box_idx]==1] = classes[box_idx]

    return cls_mask


for img_idx in total_img_ids:
    img_idx, img_name = img_idx
    if True:
        ann_ids = coco_class.getAnnIds(imgIds=[img_idx])
        ann_ids_ignore = coco_class_ignore.getAnnIds(imgIds=[img_idx])
        anns = coco_class.loadAnns(ann_ids)
        anns_ignore = coco_class_ignore.loadAnns(ann_ids_ignore)
        classes, boxes, masks, masks_ignore = [], [], [], []

        for ann, ann_ignore in zip(anns, anns_ignore):
            classes.append(ann['category_id'])
            boxes.append(ann['bbox'])
            masks.append(coco_class.annToMask(ann))
            masks_ignore.append(coco_class_ignore.annToMask(ann_ignore))

        ignore_mask = np.zeros(masks[-1].shape)

        for m, m_ignore in zip(masks, masks_ignore):
            ignore_mask[(m == 0) & (m_ignore == 1)] = 1
        total_mask = get_aggregated_mask(classes, boxes, masks)
        total_mask[ignore_mask == 1] = 255

        out = Image.fromarray(total_mask.astype(np.uint8), mode='P')
        out.putpalette(palette)
        img_idx = str(img_idx)
        out.save(os.path.join(save_dir, '%s_%s.png' % (img_idx[:4], img_idx[4:])))