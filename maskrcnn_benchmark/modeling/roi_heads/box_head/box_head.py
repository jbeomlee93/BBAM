# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels, BBAM=False):
        super(ROIBoxHead, self).__init__()
        self.BBAM = BBAM
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg, BBAM=BBAM)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, return_index=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.BBAM:
            self.return_for_BBAM = {}

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            # print(proposals)
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
            # print(proposals)

        ##

        # print(proposals)
        # print(proposals[0].fields())
        # print(proposals[0].get_field('objectness'))
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)   # ROI Pooling + 2 MLP
        # print(x.shape)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        # print(box_regression.shape)
        # print(box_regression)

        if self.BBAM:
            self.return_for_BBAM['proposals'] = proposals
            self.return_for_BBAM['class_logits'] = class_logits
            self.return_for_BBAM['box_regression'] = box_regression
        if not self.training:
            if self.BBAM:
                if return_index:
                    result, self.return_for_BBAM, selected_idxs = self.post_processor((class_logits, box_regression), proposals, self.return_for_BBAM, return_index=return_index)
                else:
                    result, self.return_for_BBAM = self.post_processor((class_logits, box_regression), proposals, self.return_for_BBAM, return_index=return_index)

            else:
                result = self.post_processor((class_logits, box_regression), proposals)

            if self.BBAM:
                if return_index:
                    return x, result, {}, self.return_for_BBAM, selected_idxs
                else:
                    return x, result, {}, self.return_for_BBAM
            else:
                return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels, BBAM=False):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels, BBAM=BBAM)
