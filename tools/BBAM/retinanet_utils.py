import os
import torch
from tools.BBAM.BBAM_utils import get_max_iou
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def get_result_single_level(anchors, box_cls, box_regression, box_coder, w, h):
    N, _, H, W = box_cls.shape
    A = box_regression.size(1) // 4  # num anchors
    C = box_cls.size(1) // A  # num class
    box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
    box_cls = box_cls.sigmoid()

    box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

    # batch is always 1
    per_box_cls = box_cls[0]  # n_anchors , n_cls

    per_box_regression = box_regression[0]
    per_anchors = anchors[0]
    detections = box_coder.decode(
        per_box_regression.view(-1, 4),
        per_anchors.bbox.view(-1, 4)
    )
    detections = BoxList(detections, (h, w), mode="xyxy")
    detections = detections.clip_to_image(remove_empty=False)
    detections = detections.bbox.view(-1, 4)
    logits, labels = per_box_cls.max(1)

    return per_anchors.bbox.view(-1, 4), logits, per_box_regression.view(-1, 4), detections, labels

def get_results_all_levels(return_for_BBAM, w, h):
    proposals = return_for_BBAM['proposals']  # [box_feat1, box_feat2, ..., box_feat5]
    class_logits = return_for_BBAM['class_logits']  # [tensor_feat1, ..., tensor_feat5], each: 1, n_cls*n_anchor, w, h
    box_regression = return_for_BBAM['box_regression']  # [tensor_feat1, ..., tensor_feat5], each: 1, n_anchor*4, w, h
    box_coder = BoxCoder(weights=(10., 10., 5., 5.))
    proposals = list(zip(*proposals))

    total_proposals = []
    total_logits = []
    total_regression = []
    total_after_regression = []
    total_labels = []

    for p, c, b in zip(proposals, class_logits, box_regression):
        single_proposals, single_logits, single_regression, single_after_regression, single_labels = get_result_single_level(p, c, b, box_coder, w, h)
        total_proposals.append(single_proposals)
        total_logits.append(single_logits)
        total_regression.append(single_regression)
        total_after_regression.append(single_after_regression)
        total_labels.append(single_labels)
        # print("single level", single_proposals.shape, single_logits.shape, single_regression.shape, single_after_regression.shape, single_labels.shape)
    total_proposals = torch.cat(total_proposals, dim=0)
    total_logits = torch.cat(total_logits, dim=0)
    total_regression = torch.cat(total_regression, dim=0)
    total_after_regression = torch.cat(total_after_regression, dim=0)
    total_labels = torch.cat(total_labels, dim=0)

    return total_proposals, total_logits, total_regression, total_after_regression, total_labels


def get_positive_idxs(logits, detections, labels, target):
    target_labels = target.get_field('labels')
    max_iou = 0
    positive_idxs = []
    # sort_flatten_logits = sorted(logits, reverse=True)
    # threshold = sort_flatten_logits[100]
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    correct_target_idx = []

    for th in thresholds:
        candidate_idxs = list(torch.where(logits>th)[0].data.cpu().numpy())
        if len(candidate_idxs) > 20:
            continue
    for c_idx in candidate_idxs:
        if labels[c_idx].cpu().numpy() not in list(target_labels.numpy()):
            candidate_idxs.remove(c_idx)

    for idx in candidate_idxs:
        iou = get_max_iou(detections[idx], target)
        if iou > max_iou:
            max_iou = iou
        if iou > 0.8:
            positive_idxs.append(idx)

    return positive_idxs

