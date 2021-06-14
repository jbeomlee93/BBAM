import numpy as np
import os
import json
import cv2
from PIL import Image
from tools.BBAM.data_utils import BBAM_data_loader
from maskrcnn_benchmark.structures.bounding_box import BoxList

import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import copy
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "plant",
        "sheep",
        "sofa",
        "train",
        "monitor",
    )

class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))
ind_to_class = dict(zip(range(len(CLASSES)), CLASSES))


def get_same_n_proposal(img_list):
    n_proposals = []
    print(n_proposals)
    for img_name in img_list:
        target = get_groundtruth(img_name)
        n_proposals.append(len(target["boxes"]))
    print(len(img_list), len(n_proposals))
    n_proposals, img_list = zip(*sorted(zip(n_proposals, img_list)))
    print(n_proposals)
    return img_list


def mask_norm(mask):
    if np.max(mask) == 0:
        return mask
    else:
        return (mask - np.min(mask)) / np.max(mask)


def get_top_CRF(mask, w, h, p_box, th_fg, postprocessor, img):
    visualize = False
    nullmask_init_prob = [0.7, 0.8, 0.9, 0.95, 0.96]

    mask = (mask - np.min(mask)) / np.max(mask)
    # print("sum", np.sum(mask))
    mask = cv2.resize(mask, (w, h))
    box_size = (p_box[3] - p_box[1]) * (p_box[2] - p_box[0])

    only_roi = np.zeros(mask.shape, dtype=np.float32)
    only_roi[p_box[1]:p_box[3], p_box[0]:p_box[2]] = 1
    mask *= only_roi
    sort_flatten_mask = sorted(mask.flatten(), reverse=True)
    threshold = sort_flatten_mask[int(box_size * (1 - th_fg))] + 1e-8
    mask = np.clip(mask / threshold, 0, 1)
    kernel_size = int(16 + 48 * np.power(box_size / (w * h), 0.5))
    mask = cv2.GaussianBlur(mask.astype(np.float64),
                            (int(kernel_size) * 2 + 1, int(kernel_size) * 2 + 1), kernel_size)
    mask *= only_roi

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = np.concatenate([np.expand_dims(mask, 0), np.expand_dims(1 - mask, 0)],
                          axis=0)
    mask = postprocessor(img, mask)[0]
    if visualize:
        cv2.imshow('CRF_mask', mask)
    total_mask = np.zeros(mask.shape)
    total_mask[mask > 0.9] = 1
    cv2.waitKey(0)
    if total_mask.sum() > box_size / 5 or float(box_size / w / h) > 1/10:
        return total_mask
    else:
        for init_prob in nullmask_init_prob:
            total_mask_start = total_mask.copy()
            total_mask_start[p_box[1]:p_box[3], p_box[0]:p_box[2]] = init_prob
            total_mask_start[total_mask == 1] = 1
            total_mask_start = np.concatenate([np.expand_dims(total_mask_start, 0), np.expand_dims(1 - total_mask_start, 0)],
                                        axis=0)
            total_mask_start = postprocessor(img, total_mask_start)[0]
            new_mask = np.zeros(total_mask_start.shape)
            new_mask[total_mask_start > 0.9] = 1
            if new_mask.sum() > box_size / 3:
                return new_mask

        return new_mask




def get_final_mask_with_topCRF(image_name, mask_dir, img, th_fg):
    return_dict = {}
    target = get_groundtruth(image_name)
    mask_dir = os.path.join(mask_dir, image_name)  # pidx_0000_cls, pidx_0001_cls, ...
    w, h = target["img_size"]

    nullmask_init_prob = [0.7, 0.8, 0.9, 0.95]
    return_dict['score'], return_dict['mask'], return_dict['class'], return_dict['box'] = [], [], [], []

    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=80,
        bi_rgb_std=3,
        bi_w=4,
    )
    for p_idx, p_label in enumerate(target["labels"]):
        p_box = target["boxes"][p_idx]
        mask = cv2.imread(os.path.join(mask_dir, 'pidx_%04d_%s' % (p_idx, ind_to_class[p_label]), 'iter_0299_mask.jpg'))
        return_dict['score'].append(1)
        return_dict['class'].append(target["labels"][p_idx])
        return_dict['box'].append(p_box)

        if (mask is None):
            for init_prob in nullmask_init_prob:
                total_mask = np.zeros((h, w))
                total_mask[p_box[1]:p_box[3], p_box[0]:p_box[2]] = init_prob
                total_mask = np.concatenate([np.expand_dims(total_mask, 0), np.expand_dims(1 - total_mask, 0)],
                                      axis=0)
                total_mask = postprocessor(img, total_mask)[0]
                box_area = float((p_box[2]-p_box[0]) * (p_box[3]-p_box[1]))
                new_mask = np.zeros(total_mask.shape)
                new_mask[total_mask > 0.9] = 1
                if new_mask.sum() > box_area / 5:
                    return_dict['mask'].append(new_mask)
                    break
            continue
        mask = np.array(mask[:, :, 0], dtype=np.float32) / 255.0

        if np.max(mask) == 0:  # mask exists, but all 0
            for init_prob in nullmask_init_prob:
                total_mask = np.zeros((h, w))
                total_mask[p_box[1]:p_box[3], p_box[0]:p_box[2]] = init_prob
                total_mask = np.concatenate([np.expand_dims(total_mask, 0), np.expand_dims(1 - total_mask, 0)],
                                            axis=0)
                total_mask = postprocessor(img, total_mask)[0]
                box_area = float((p_box[2]-p_box[0]) * (p_box[3]-p_box[1]))
                new_mask = np.zeros(total_mask.shape)
                new_mask[total_mask > 0.9] = 1
                if new_mask.sum() > box_area / 5:
                    return_dict['mask'].append(new_mask)
                    break

        else:
            total_mask = get_top_CRF(mask, w, h, p_box, th_fg, postprocessor, img)

            return_dict['mask'].append(total_mask)

    if len(return_dict['mask']) == 0:
        print("exists?????????????")
        return return_dict
    else:
        return_dict['score'] = np.array(return_dict['score'])
        return_dict['class'] = np.array(return_dict['class'])
        return_dict['mask'] = np.concatenate([m[np.newaxis, :, :] for m in return_dict['mask']], axis=0)
    return return_dict


def get_groundtruth(img_id):
    annopath = os.path.join("Dataset/VOC2012_SEG_AUG", "Annotations", "%s.xml")
    anno = ET.parse(annopath % img_id).getroot()
    anno = preprocess_annotation(anno)
    height, width = anno["im_info"]

    anno["img_size"] = (width, height)
    return anno

def preprocess_annotation(target):
    boxes = []
    gt_classes = []
    difficult_boxes = []
    TO_REMOVE = 1

    for obj in target.iter("object"):
        difficult = int(obj.find("difficult").text) == 1
        if not True and difficult:
            continue
        name = obj.find("name").text.lower().strip()
        bb = obj.find("bndbox")
        # Make pixel indexes 0-based
        # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
        box = [
            bb.find("xmin").text,
            bb.find("ymin").text,
            bb.find("xmax").text,
            bb.find("ymax").text,
        ]
        bndbox = tuple(
            map(lambda x: x - TO_REMOVE, list(map(int, box)))
        )

        boxes.append(bndbox)
        if name == 'tvmonitor':
            name='monitor'
        if name == 'diningtable':
            name='table'
        if name == 'pottedplant':
            name='plant'
        gt_classes.append(class_to_ind[name])
        difficult_boxes.append(difficult)

    size = target.find("size")
    im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

    res = {
        "boxes": boxes,
        "labels": gt_classes,
        "difficult": difficult_boxes,
        "im_info": im_info,
    }
    return res

#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def get_groundtruth_coco(coco_class, img_id):
    anno_ids = coco_class.getAnnIds(imgIds=int(img_id))
    anno = coco_class.loadAnns(anno_ids)
    match_class = class_matching(coco_class)

    anno = _preprocess_annotation_coco(anno, img_id, match_class, coco_class)
    height, width = anno["im_info"]
    anno["img_size"] = (width, height)

    return anno

def _preprocess_annotation_coco(target, img_id, match_class, coco_class):
    boxes = []
    gt_classes = []
    TO_REMOVE = 1
    for obj in target:
        box = obj['bbox']
        box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        bndbox = tuple(
            map(lambda x: x - TO_REMOVE, list(map(int, box)))
        )
        boxes.append(bndbox)
        gt_classes.append(match_class[obj['category_id']])
    im_info = coco_class.imgs[int(img_id)]
    im_info = tuple(map(int, (im_info['height'], im_info['width'])))
    res = {
        "boxes": boxes,
        "labels": gt_classes,
        "im_info": im_info,
    }
    return res

def class_matching(coco_class):
    cats_original = coco_class.cats
    norm_cats = {0:0}
    idx = 1
    for cat in cats_original:
        norm_cats[cat] = idx
        idx += 1

    return norm_cats