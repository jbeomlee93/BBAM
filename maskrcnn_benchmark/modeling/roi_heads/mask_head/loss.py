# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

import cv2
import numpy as np
import os

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        # print("mask", segmentation_mask)
        # print("proposal", proposal)
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        # print("mask", mask.shape)
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, outroot):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.outroot = outroot
        self.iter_mask_loss = 0

    def build_tfboard_writer(self, device):
        print(device)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        # print(proposals, targets)
        for proposals_per_image, targets_per_image in zip(proposals, targets): # each batch?
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def __call__(self, proposals, mask_logits, targets, target_ignore=None, nega_mask_logits=None, nega_coeff=0):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """

        total_seed_growing = True

        # print(targets, target_ignore)
        labels, mask_targets = self.prepare_targets(proposals, targets)
        if not target_ignore is None:
            _, mask_targets_ignore = self.prepare_targets(proposals, target_ignore)
        # print(len(mask_targets), len(mask_targets_ignore))
        # print(mask_targets[0].shape, mask_targets[1].shape, mask_targets_ignore[0].shape, mask_targets_ignore[1].shape)
        # for m_target, m_target_ignore in zip(mask_targets, mask_targets_ignore):
        #     for m_t, m_t_i in zip(m_target, m_target_ignore):
        #         self.visualize_mask(m_t, m_t_i)
        #
        # # print(mask_targets, mask_targets_ignore)
        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)
        is_mismatch = False
        if not target_ignore is None:
            mask_targets_ignore = cat(mask_targets_ignore, dim=0)
            # print( mask_targets.shape, mask_targets_ignore.shape)
            if mask_targets.shape != mask_targets_ignore.shape:
                print("mismatch")
                is_mismatch = True
        # print(mask_targets.shape, mask_targets_ignore.shape)
        # print(mask_targets, mask_targets_ignore)
        # print(labels)
        # print(mask_targets)
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        # print(mask_logits[positive_inds, labels_pos].shape, mask_targets.shape, mask_targets_ignore.shape)

        loss_balance = True
        if target_ignore is None or is_mismatch:
            if loss_balance:
                mask_loss_structure = F.binary_cross_entropy_with_logits(
                    mask_logits[positive_inds, labels_pos], mask_targets, reduction='none'
                )
                mask_loss = (mask_loss_structure[mask_targets == 1]).mean() + (mask_loss_structure[mask_targets == 0]).mean()
                mask_loss /= 2

            else:
                mask_loss = F.binary_cross_entropy_with_logits(
                    mask_logits[positive_inds, labels_pos], mask_targets
                )

        else:
            seed_growing = True

            if not seed_growing:
                mask_logit = mask_logits[positive_inds, labels_pos]
                mask_loss_structure = F.binary_cross_entropy_with_logits(
                    mask_logit, mask_targets, reduction='none'
                )
                ignore_mask = torch.ones_like(mask_loss_structure)

                for m_idx, (m_target, m_target_ignore) in enumerate(zip(mask_targets, mask_targets_ignore)):
                    if torch.all(torch.eq(m_target, m_target_ignore)):  # ignore all
                        ignore_mask[m_idx, :, :] = 0
                    else:
                        ignore_mask[m_idx][(m_target == 0) & (m_target_ignore == 1)] = 0

                total_pixels = ignore_mask.sum()

            else:
                th_growing = 0.9
                # print("th_growing", th_growing)
                mask_logit = mask_logits[positive_inds, labels_pos]  # [5, 28, 28]

                fg_select = mask_targets.clone()  # minimal foreground, 1: fg
                bg_select = (1 - mask_targets_ignore).clone()  # minimal background, 1: bg

                initial_seed_fg = (fg_select == 1)
                initial_seed_bg = (bg_select == 1)
                before_grown_fg = fg_select.sum()
                before_grown_bg = bg_select.sum()

                mask_logit_sigmoid = torch.nn.functional.sigmoid(mask_logit)
                fg_select[mask_logit_sigmoid > th_growing] = 1  # original fg, grown fg = 1
                bg_select[mask_logit_sigmoid < (1 - th_growing)] = 1  # original bg, grown bg = 1

                # fg_select and bg_select may be both 1, if:
                # initial seed: foreground, but mask_logit < 1-th_growing
                fg_select[initial_seed_bg] = 0
                bg_select[initial_seed_fg] = 0

                # now, fg_select and bg_select: disjoint

                W = torch.ones([1, 1, 3, 3]).to(fg_select.device)
                W[:, :, 1, 1] = 0  # consider only neighbors
                fg_select_neighbor = torch.nn.functional.conv2d(fg_select.unsqueeze(1), W, stride=1, padding=1)[:, 0]
                bg_select_neighbor = torch.nn.functional.conv2d(bg_select.unsqueeze(1), W, stride=1, padding=1)[:, 0]

                mask_target_news = torch.zeros_like(mask_targets) - 1
                mask_target_news[(fg_select == 1) & (fg_select_neighbor > 0)] = 1
                mask_target_news[(bg_select == 1) & (bg_select_neighbor > 0)] = 0
                # print(((mask_target_news==1).sum()/before_grown_fg).data.cpu().numpy())
                total = float((mask_logit_sigmoid>0).sum())
                fg = float((mask_logit_sigmoid>th_growing).sum())
                bg = float((mask_logit_sigmoid<1-th_growing).sum())
                # print("over", fg/total, bg/total)

                ignore_mask = torch.zeros_like(mask_target_news)
                ignore_mask[mask_target_news > -1] = 1
                # mask_target_news[mask_target_news == -1] = 0

                mask_loss_structure = F.binary_cross_entropy_with_logits(
                    mask_logit, mask_target_news, reduction='none'
                )

                total_pixels = ignore_mask.sum()
                # print(ignore_mask.mean().data.cpu().numpy())

            if total_pixels == 0:
                mask_loss = (mask_loss_structure * ignore_mask).sum() * 0
            else:
                if loss_balance:
                    if seed_growing:
                        m1 = (mask_loss_structure[(mask_target_news == 1) & (ignore_mask == 1)])
                        m2 = (mask_loss_structure[(mask_target_news == 0) & (ignore_mask == 1)])
                        if m1.shape[0] == 0 or m2.shape[0] == 0:
                            print("nan")
                            mask_loss = (m1.sum()+m2.sum())*0
                        else:
                            mask_loss = m1.mean()+m2.mean()
                    else:
                        mask_loss = (mask_loss_structure[(mask_targets == 1) & (ignore_mask == 1)]).mean() \
                                    + (mask_loss_structure[(mask_targets == 0) & (ignore_mask == 1)]).mean()
                    mask_loss /= 2

                else:
                    mask_loss = (mask_loss_structure * ignore_mask).sum() / total_pixels
                    # print(mask_loss)


            # original
            """
            mask_loss_structure = F.binary_cross_entropy_with_logits(
                mask_logits[positive_inds, labels_pos], mask_targets, reduction='none'
            )
            ignore_mask = torch.ones_like(mask_loss_structure)
            for m_idx, (m_target, m_target_ignore) in enumerate(zip(mask_targets, mask_targets_ignore)):
                if torch.all(torch.eq(m_target, m_target_ignore)): # ignore all
                    ignore_mask[m_idx, :, :] = 0
                else:
                    ignore_mask[m_idx][(m_target == 0) & (m_target_ignore == 1)] = 0
            total_pixels = ignore_mask.sum()
            if total_pixels == 0:
                mask_loss = (mask_loss_structure * ignore_mask).sum() * 0
            else:
                mask_loss = (mask_loss_structure * ignore_mask).sum() / total_pixels
            """
            # print("SUM", (mask_loss_structure * ignore_mask).sum(), mask_loss_structure.sum())
            # print("MEAN", mask_loss, mask_loss_structure.mean())
        # for m_target, m_target_ignore, m_ignore in zip(mask_targets, mask_targets_ignore, ignore_mask):
        #     self.visualize_mask(m_target, m_target_ignore, m_ignore)
        # print(mask_loss.detach().cpu().numpy())
        if nega_mask_logits is None:
            self.iter_mask_loss += 1

            return mask_loss
        else:
            values, _ = torch.max(nega_mask_logits[:, 1:, :, :], dim=1)
            # print(values.shape)
            # print(mask_loss)
            # print(F.binary_cross_entropy_with_logits(
            #     values, torch.zeros_like(values)))
            nega_loss = nega_coeff * F.binary_cross_entropy_with_logits(
                values, torch.zeros_like(values))
            mask_loss += nega_loss
            self.iter_mask_loss += 1

            return mask_loss

    def visualize_mask(self, target, target_ignore, ignore_mask):
        target, target_ignore, ignore_mask = target.data.cpu().numpy().astype(np.float32), target_ignore.data.cpu().numpy().astype(np.float32), ignore_mask.data.cpu().numpy().astype(np.float32)
        cv2.imshow('target', target)
        cv2.imshow('target_ignore', target_ignore)
        cv2.imshow('ignore_mask', ignore_mask)
        diff = target_ignore-target
        print(diff.shape, np.sum((diff>=0)))
        cv2.imshow('diff', diff)
        cv2.waitKey(0)


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION, outroot=cfg.OUTPUT_DIR
    )

    return loss_evaluator
