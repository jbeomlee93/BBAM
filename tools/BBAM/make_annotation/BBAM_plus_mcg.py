import json
from pycocotools.mask import decode
from pycocotools import mask as maskUtils
import cv2
import numpy as np
import os
from scipy import io
import time
import joblib
from pycococreatortools import create_image_info, create_annotation_info
import torch
import random
from pycocotools import coco


def annToRLE(ann, pred=False):
    if pred:
        h, w = ann['size'][0], ann['size'][1]
        segm = ann
    else:
        h, w = ann['height'], ann['width']
        segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)

    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, pred=False):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, pred)
    m = maskUtils.decode(rle)
    return m

def binary_iou(A, B):
    intersection = np.sum(np.where((A==1) & (B==1)))
    union = np.sum(np.where((A==1) | (B==1)))

    return intersection/union

def check_mcg_in_box(A, box):
    box = (int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]))
    box_area = np.zeros_like(A)
    box_area[box[1]:box[3], box[0]:box[2]] = 1
    total_pixel = np.sum(np.where(A==1))
    total_pixel_in_box = np.sum(np.where((A==1) & (box_area==1)))
    if total_pixel_in_box/total_pixel > 0.95:
        return True
    else:
        return False

def check_mcg_in_BBAM(A, B, th=0.95):
    # A:mcg proposal, B: BBAM mask
    intersection = np.sum(np.where((A==1) & (B==1)))
    # print(intersection / np.sum(np.where(A==1)))
    if intersection / np.sum(np.where(A==1)) > th:
        return True
    else:
        return False


def push_box(box, superpixels, d=5):
    h, w = superpixels.shape

    new_box = [0,0,0,0]
    new_box[0] = max(box[0]-d, 0)
    new_box[1] = max(box[1]-d, 0)
    new_box[2] = min(box[2]+d, w)
    new_box[3] = min(box[3]+d, h)

    return box


def filter_super_pixels(mat_file, box):
    box = (int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3]))
    superpixels = mat_file['superpixels'].copy()
    box = push_box(box, superpixels)
    superpixels[box[1]:box[3], box[0]:box[2]] = 0
    out_list = list(np.unique(superpixels))[1:]

    superpixels = mat_file['superpixels'].copy()
    sup_temp = np.zeros_like(superpixels)
    sup_temp[box[1]:box[3], box[0]:box[2]] = superpixels[box[1]:box[3], box[0]:box[2]]
    in_list = list(np.unique(sup_temp))

    out_list = list(set(out_list) - set(in_list))

    filter_list = []
    for s in mat_file['labels']:
        if np.intersect1d(np.array(out_list), np.array(s[0][0])).shape[0] == 0:
            filter_list.append(s)
    return filter_list

def match_mcg_proposal_single(BBAM, file_name, bbox):
    if bbox[2] * bbox[3] < 32*32:
        return BBAM, True

    # mat_dir = 'Dataset/MCG-Pascal-Main_trainvaltest_2012-proposals/'
    # mat_dir_2007 = 'Dataset/MCG-Pascal-Main_trainvaltest_2007-proposals/'
    if '2007_' in file_name:
        mat_file = io.loadmat(os.path.join(mat_dir_2007, file_name.replace('2007_', '').replace('jpg', 'mat')))
    else:
        mat_file = io.loadmat(os.path.join(mat_dir, file_name.replace('jpg', 'mat')))

    best_overlap = 0
    selected_proposal = None
    filtered_labels = filter_super_pixels(mat_file, bbox)
    # filtered_labels = mat_file['labels']
    superpixels = mat_file['superpixels']
    selected_proposal_add = np.zeros_like(superpixels)

    for s in filtered_labels:
        A = np.zeros_like(superpixels)
        # print(s[0][0])
        data = s[0][0]
        spl = [0] + [i for i in range(1, len(data)) if data[i] - data[i - 1] > 1] + [None]
        data = [data[b:e] for (b, e) in [(spl[i - 1], spl[i]) for i in range(1, len(spl))]]

        for ss in data:
            start, end = ss[0], ss[-1]
            A[(start <= superpixels) & (superpixels <= end)] = 1
        # for ss in s[0][0]:
        #     A[superpixels == ss] = 1

        iou = binary_iou(A, BBAM)
        if iou > best_overlap and check_mcg_in_box(A, bbox):
            selected_proposal = A
            best_overlap = iou

        # perfectly inside in BBAM
        if check_mcg_in_BBAM(A, BBAM) and check_mcg_in_BBAM(A, bbox):
            selected_proposal_add[A==1] = A[A==1]

    if selected_proposal is None:
        return BBAM, True
    else:
        if np.max(selected_proposal_add) > 0:
            selected_proposal[selected_proposal_add==1] = selected_proposal_add[selected_proposal_add==1]
        # cv2.imshow('selected_add', (selected_proposal*255.0).astype('uint8'))
        # cv2.waitKey(0)
        only_roi = np.zeros(selected_proposal.shape, dtype=np.float32)
        p_box = (int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        only_roi[p_box[1]:p_box[3], p_box[0]:p_box[2]] = 1
        return selected_proposal*only_roi, False

def match_mcg_proposal(BBAM1, BBAM2, file_name, bbox1, bbox2):
    if bbox1[2] * bbox1[3] < 32*32:
        return BBAM1, BBAM2, True, True
    # mat_dir = 'Dataset/MCG-Pascal-Main_trainvaltest_2012-proposals/'
    # mat_dir_2007 = 'Dataset/MCG-Pascal-Main_trainvaltest_2007-proposals/'
    if '2007_' in file_name:
        mat_file = io.loadmat(os.path.join(mat_dir_2007, file_name.replace('2007_', '').replace('jpg', 'mat')))
    else:
        mat_file = io.loadmat(os.path.join(mat_dir, file_name.replace('jpg', 'mat')))

    best_overlap1, best_overlap2 = 0, 0
    selected_proposal1, selected_proposal2 = None, None
    filtered_labels = filter_super_pixels(mat_file, bbox1)
    # filtered_labels = mat_file['labels']
    superpixels = mat_file['superpixels']
    selected_proposal_add1, selected_proposal_add2 = np.zeros_like(superpixels), np.zeros_like(superpixels)

    for s in filtered_labels:
        A = np.zeros_like(superpixels)
        # print(s[0][0])
        data = s[0][0]
        spl = [0] + [i for i in range(1, len(data)) if data[i] - data[i - 1] > 1] + [None]
        data = [data[b:e] for (b, e) in [(spl[i - 1], spl[i]) for i in range(1, len(spl))]]

        for ss in data:
            start, end = ss[0], ss[-1]
            A[(start <= superpixels) & (superpixels <= end)] = 1
        # for ss in s[0][0]:
        #     A[superpixels == ss] = 1

        iou1 = binary_iou(A, BBAM1)
        iou2 = binary_iou(A, BBAM2)
        if iou1 > best_overlap1 and check_mcg_in_box(A, bbox1):
            selected_proposal1 = A.copy()
            best_overlap1 = iou1
        if iou2 > best_overlap2 and check_mcg_in_box(A, bbox2):
            selected_proposal2 = A.copy()
            best_overlap2 = iou2
            # cv2.imshow('best', (A * 255.0).astype('uint8'))
            # cv2.waitKey(0)
        # perfectly inside in BBAM

        if check_mcg_in_BBAM(A, BBAM1):
            selected_proposal_add1[A==1] = A[A==1]
        if check_mcg_in_BBAM(A, BBAM2):
            selected_proposal_add2[A==1] = A[A==1]
            # cv2.imshow('super', (selected_proposal_add2 * 255.0).astype('uint8'))
            # cv2.waitKey(0)
    flag1, flag2 = False, False
    if selected_proposal1 is None:
        return1 = BBAM1
        flag1 = True


    else:
        only_roi = np.zeros(selected_proposal1.shape, dtype=np.float32)
        p_box = (int(bbox1[0]), int(bbox1[1]), int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))

        only_roi[p_box[1]:p_box[3], p_box[0]:p_box[2]] = 1
        if np.max(selected_proposal_add1) > 0:
            selected_proposal1[selected_proposal_add1==1] = selected_proposal_add1[selected_proposal_add1==1]

        return1 = selected_proposal1 * only_roi

    if selected_proposal2 is None:
        return2 = BBAM2
        flag2 = True
    else:
        only_roi = np.zeros(selected_proposal2.shape, dtype=np.float32)
        p_box = (int(bbox2[0]), int(bbox2[1]), int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))

        only_roi[p_box[1]:p_box[3], p_box[0]:p_box[2]] = 1
        if np.max(selected_proposal_add2) > 0:
            selected_proposal2[selected_proposal_add2 == 1] = selected_proposal_add2[selected_proposal_add2 == 1]
        if selected_proposal1 is None:
            selected_proposal2[BBAM1 == 1] = BBAM1[BBAM1 == 1]
        else:
            selected_proposal2[selected_proposal1 == 1] = selected_proposal1[selected_proposal1 == 1]

        return2 = selected_proposal2 * only_roi

    return return1, return2, flag1, flag2

# def process()



json_file_name1 = 'datasets/BBAM_training_images_nimg_10582_th_0.85.json'
json_file_name2 = 'datasets/BBAM_training_images_nimg_10582_th_0.2.json'

json_file1 = json.load(open(json_file_name1))
# json_file2 = json.load(open(json_file_name2))
coco_class1 = coco.COCO(annotation_file=json_file_name1)
coco_class2 = coco.COCO(annotation_file=json_file_name2)
coco_output1 = {}
coco_output1["images"] = []
coco_output1["annotations"] = []
coco_output1['categories'] = json_file1['categories']
coco_output1['type'] = json_file1['type']

coco_output2 = {}
coco_output2["images"] = []
coco_output2["annotations"] = []
coco_output2['categories'] = json_file1['categories']
coco_output2['type'] = json_file1['type']


chunks = []
n_jobs = 100


mat_dir = 'Dataset/MCG-Pascal-Main_trainvaltest_2012-proposals/'
mat_dir_2007 = 'Dataset/MCG-Pascal-Main_trainvaltest_2007-proposals/'

def process(json_file_images_i, js_1, js_2):
    img_id = json_file_images_i['id']
    annotation_info_fgs1 = []
    annotation_info_fgs2 = []
    print(img_id)

    if len(js_1) == len(js_2):
        for j_idx, j1 in enumerate(js_1):
            if (not os.path.exists(os.path.join(mat_dir, json_file_images_i['file_name'].replace('jpg', 'mat'))))\
                    and (not os.path.exists(os.path.join(mat_dir_2007, json_file_images_i['file_name'].replace('2007_', '').replace('jpg', 'mat')))):
                print(json_file_images_i['file_name'], "not exist")
                annotation_info_fgs1.append(js_1[j_idx])
                annotation_info_fgs2.append(js_2[j_idx])

            else:
                mask1 = annToMask(js_1[j_idx])
                mask2 = annToMask(js_2[j_idx])
                refine_mask1, refine_mask2, flag1, flag2 = match_mcg_proposal(mask1, mask2, json_file_images_i['file_name'], js_1[j_idx]['bbox'], js_2[j_idx]['bbox'])
                category_info1 = {'id': int(js_1[j_idx]['category_id']), 'is_crowd': False}
                category_info2 = {'id': int(js_2[j_idx]['category_id']), 'is_crowd': False}
                tol = 0 if flag1 else 2
                annotation_info_fg = create_annotation_info(
                    js_1[j_idx]['id'], js_1[j_idx]['image_id'], category_info1, refine_mask1, [js_1[j_idx]['width'], js_1[j_idx]['height']], tolerance=tol,
                    bounding_box=torch.Tensor(js_1[j_idx]['bbox']))
                if annotation_info_fg is None:
                    annotation_info_fgs1.append(js_1[j_idx])
                else:
                    annotation_info_fgs1.append(annotation_info_fg)
                tol = 0 if flag2 else 2
                annotation_info_fg = create_annotation_info(
                    js_2[j_idx]['id'], js_2[j_idx]['image_id'], category_info2, refine_mask2,
                    [js_2[j_idx]['width'], js_2[j_idx]['height']], tolerance=tol,
                    bounding_box=torch.Tensor(js_2[j_idx]['bbox']))
                if annotation_info_fg is None:
                    annotation_info_fgs2.append(js_2[j_idx])
                else:
                    annotation_info_fgs2.append(annotation_info_fg)

    else:
        for j_idx, j in enumerate(js_1):
            if (not os.path.exists(os.path.join(mat_dir, json_file_images_i['file_name'].replace('jpg', 'mat')))) \
                    and (not os.path.exists(os.path.join(mat_dir_2007, json_file_images_i['file_name'].replace('2007_', '').replace('jpg', 'mat')))):
                print(json_file_images_i['file_name'], "not exist")
                annotation_info_fgs1.append(j)
            else:
                mask = annToMask(j)
                refine_mask, flag = match_mcg_proposal_single(mask, json_file_images_i['file_name'], j['bbox'])
                category_info = {'id': int(j['category_id']), 'is_crowd': False}
                tol = 0 if flag else 2
                annotation_info_fg = create_annotation_info(
                    j['id'], j['image_id'], category_info, refine_mask, [j['width'], j['height']], tolerance=tol,
                    bounding_box=torch.Tensor(j['bbox']))
                if annotation_info_fg is None:
                    annotation_info_fgs1.append(js_1[j_idx])
                else:
                    annotation_info_fgs1.append(annotation_info_fg)

        for j_idx, j in enumerate(js_2):
            if (not os.path.exists(os.path.join(mat_dir, json_file_images_i['file_name'].replace('jpg', 'mat')))) \
                    and (not os.path.exists(os.path.join(mat_dir_2007, json_file_images_i['file_name'].replace('2007_', '').replace('jpg', 'mat')))):
                print(json_file_images_i['file_name'], "not exist")
                annotation_info_fgs2.append(j)
            else:
                mask = annToMask(j)
                refine_mask, flag = match_mcg_proposal_single(mask, json_file_images_i['file_name'], j['bbox'])
                category_info = {'id': int(j['category_id']), 'is_crowd': False}
                tol = 0 if flag else 2

                annotation_info_fg = create_annotation_info(
                    j['id'], j['image_id'], category_info, refine_mask, [j['width'], j['height']], tolerance=tol,
                    bounding_box=torch.Tensor(j['bbox']))
                if annotation_info_fg is None:
                    annotation_info_fgs2.append(js_2[j_idx])
                else:
                    annotation_info_fgs2.append(annotation_info_fg)
    return json_file_images_i, annotation_info_fgs1, annotation_info_fgs2


total_list = coco_class1.getImgIds()
json_imgs = coco_class1.loadImgs(total_list)


results = joblib.Parallel(n_jobs=n_jobs, verbose=50, pre_dispatch="all")(
    [joblib.delayed(process)(coco_class1.loadImgs(total_list[i])[0], coco_class1.loadAnns(coco_class1.getAnnIds(total_list[i])),
                                       coco_class2.loadAnns(coco_class2.getAnnIds(total_list[i]))) for i in range(len(total_list))])
for ann_idx, ann in enumerate(results):
    img_info, annotation_info_fgs1, annotation_info_fgs2 = ann
    coco_output1["images"].append(img_info)
    coco_output2["images"].append(img_info)

    for ann_info_idx, annotation_info_fg in enumerate(annotation_info_fgs1):
        if annotation_info_fg == None:
            continue
        coco_output1['annotations'].append(annotation_info_fg)
    for ann_info_idx, annotation_info_fg in enumerate(annotation_info_fgs2):
        if annotation_info_fg == None:
            continue
        coco_output2['annotations'].append(annotation_info_fg)

with open(json_file_name1.replace('.json', '_plus_mcg.json'), 'w') as outfile:
    json.dump(coco_output1, outfile)
with open(json_file_name2.replace('.json', '_plus_mcg.json'), 'w') as outfile:
    json.dump(coco_output2, outfile)


