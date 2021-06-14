# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, BBAM=False):
        super(GeneralizedRCNN, self).__init__()
        self.BBAM = BBAM

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels, BBAM=self.BBAM)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels, BBAM=self.BBAM)
        self.cfg = cfg.clone()
    def forward(self, images, targets=None, fixed_proposals=None, target_ignore=None, return_index=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if self.cfg.MODEL.RETINANET_ON and self.BBAM:
            proposals, proposal_losses, return_for_BBAM = self.rpn(images, features, targets)
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)

        # print("proposals", proposals)
        # if self.BBAM:
        #     print(proposals)

        # retinanet: no roi heads, all done in RPN head
        if self.roi_heads:
            if self.BBAM:
                if fixed_proposals:
                    if return_index:
                        x, result, detector_losses, return_for_BBAM, return_idx_for_BBAM = self.roi_heads(features, fixed_proposals, targets, return_index=return_index)
                    else:
                        x, result, detector_losses, return_for_BBAM = self.roi_heads(features, fixed_proposals, targets)

                else:
                    if return_index:
                        x, result, detector_losses, return_for_BBAM, return_idx_for_BBAM = self.roi_heads(features, proposals, targets, return_index=return_index)
                    else:
                        x, result, detector_losses, return_for_BBAM = self.roi_heads(features, proposals, targets)


            else:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, target_ignore=target_ignore)


        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        if self.BBAM:
            if return_index:
                return result, return_for_BBAM, return_idx_for_BBAM
            else:
                return result, return_for_BBAM
        else:
            return result