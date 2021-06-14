# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def get_max_iou(sources, targets):
    # target: multiple boxes

    maxIoUs = torch.zeros_like(sources.get_field("labels")).type(torch.float32)

    for s_idx, source in enumerate(sources.bbox):
        maxIoU = 0
        for target in targets.bbox:
            bb1, bb2 = {}, {}
            bb1['x1'], bb1['x2'] = int(source[0]), int(source[2])
            bb1['y1'], bb1['y2'] = int(source[1]), int(source[3])
            bb2['x1'], bb2['x2'] = int(target[0]), int(target[2])
            bb2['y1'], bb2['y2'] = int(target[1]), int(target[3])
            # determine the coordinates of the intersection rectangle
            x_left = max(bb1['x1'], bb2['x1'])
            y_top = max(bb1['y1'], bb2['y1'])
            x_right = min(bb1['x2'], bb2['x2'])
            y_bottom = min(bb1['y2'], bb2['y2'])

            if x_right < x_left or y_bottom < y_top:
                continue

            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
            bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            assert iou >= 0.0
            assert iou <= 1.0
            if maxIoU < iou:
                maxIoU = iou
        maxIoUs[s_idx] = maxIoU
    return maxIoUs

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        # print(labels)
        inds_mask = labels > 0
        # print(inds_mask)
        inds = inds_mask.nonzero().squeeze(1)
        # print("inds", inds)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

def keep_only_negative_boxes(boxes, targets, n_sample=None):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    # assert boxes[0].has_field("labels")
    negative_boxes = []
    negative_inds = []
    num_boxes = 0
    for b_idx, boxes_per_image in enumerate(boxes):
        max_ious = get_max_iou(boxes_per_image, targets[b_idx])
        if n_sample is None:
            inds_mask = (max_ious < 0.1) & (max_ious > 0)
            inds = inds_mask.nonzero().squeeze(1)

        else:
            max_ious[max_ious>0.1] = 0
            inds_mask = (max_ious < 0.1) & (max_ious > 0)
            # print(inds_mask.nonzero().squeeze(1).shape[0], n_sample[b_idx])
            if inds_mask.nonzero().squeeze(1).shape[0] > n_sample[b_idx]:
                _, inds = torch.topk(max_ious, k=n_sample[b_idx], dim=0)
                # print("inds_here", inds)
            else:
                inds = inds_mask.nonzero().squeeze(1)

        negative_boxes.append(boxes_per_image[inds])
        negative_inds.append(inds_mask)
    return negative_boxes, negative_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, target_ignore=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
            target_ignore: ignored mask
        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # _ = keep_only_positive_boxes(proposals)

        nega = False

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
            if nega:
                n_sample = [len(p.bbox) for p in proposals]
                nega_boxes, nega_inds = keep_only_negative_boxes(all_proposals, targets, n_sample=n_sample)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
            if self.training and nega:
                x_nega = self.feature_extractor(features, nega_boxes)
        mask_logits = self.predictor(x)
        if self.training and nega:
            mask_logits_nega = self.predictor(x_nega)
            # print(mask_logits.shape, mask_logits_nega.shape)
        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            # result: BoxList
            return x, result, {}
        if nega:
            loss_mask = self.loss_evaluator(proposals, mask_logits, targets, target_ignore=target_ignore, nega_mask_logits=mask_logits_nega, nega_coeff=0.5)
        else:
            loss_mask = self.loss_evaluator(proposals, mask_logits, targets, target_ignore=target_ignore)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
