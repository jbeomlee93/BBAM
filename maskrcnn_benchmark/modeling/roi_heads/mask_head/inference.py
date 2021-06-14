# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import joblib
import cv2


# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)

        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)

        return results


class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import pycocotools.mask as mask_util
        import numpy as np

        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field("mask").cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            result.add_field("mask", rles)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))

    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale

def norm_image(img):
    # print(img.shape)
    img_new = np.zeros(img.shape)
    img_new[:] = img[:]
    for i in range(3):
        img_new[:, :, i] = img_new[:, :, i] - img_new[:, :, i].min()

        img_new[:, :, i] = img_new[:, :, i] / img_new[:, :, i].max()  * 255.0
    # img = img - img.min()
    # img = img / img.max()
    return (img_new).astype('uint8')

def crop_city_roi(p_box, img, crop_size=500, mask=None, return_ratio=False):
    # h, w max -> 250 img resize
    # p_box[0], p_box[2] -> 2048
    # cv2.imshow('mask_orig', mask)
    # print(p_box)
    h, w = p_box[2]-p_box[0], p_box[3]-p_box[1]
    if h * w < 100:
        resize_ratio = 50 / h if h > w else 50 / w

    else:
        resize_ratio = 250 / h if h > w else 250 / w
    img_resize = cv2.resize(img, dsize=None, fx=resize_ratio, fy=resize_ratio)
    if mask is not None:
        mask_resize = cv2.resize(mask, dsize=None, fx=resize_ratio, fy=resize_ratio)
    print(p_box)
    p_box = np.array(p_box) * resize_ratio
    p_box = [int(p_box[0]),int(p_box[1]),int(p_box[2]),int(p_box[3])]

    h_img, w_img = img_resize.shape[:2]
    # print(h_img, w_img)
    # print(p_box[0], crop_size/2)
    # print(max(0, p_box[0]-crop_size/2))
    cut_point_crop = [max(0, p_box[0]-crop_size/2), max(0, p_box[1]-crop_size/2),
                      min(w_img-1, p_box[2]+crop_size/2), min(h_img-1, p_box[3]+crop_size/2)]
    cut_point_crop = [int(ii) for ii in cut_point_crop]
    # print([cut_point_crop[1],cut_point_crop[3], cut_point_crop[0],cut_point_crop[2]])
    img_crop = img_resize[cut_point_crop[1]:cut_point_crop[3], cut_point_crop[0]:cut_point_crop[2]]
    if mask is not None:
        mask_crop = mask_resize[cut_point_crop[1]:cut_point_crop[3], cut_point_crop[0]:cut_point_crop[2]]

    p_box_crop = [p_box[0]-cut_point_crop[0], p_box[1]-cut_point_crop[1],
                p_box[2]-cut_point_crop[0], p_box[3]-cut_point_crop[1]]
    if mask is None:
        if return_ratio:
            return img_crop, cut_point_crop, img_resize, p_box_crop, resize_ratio
        else:
            return img_crop, cut_point_crop, img_resize, p_box_crop

    else:
        if return_ratio:
            return img_crop, cut_point_crop, img_resize, p_box_crop, mask_crop, resize_ratio
        else:
            return img_crop, cut_point_crop, img_resize, p_box_crop, mask_crop

def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1, do_CRF=False, crf_postprocessor=None, img_single=None, score=None, upscale=1.0, img_id=0):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    # print("box score", score)
    mask = mask.float()
    CRF_normalize = True
    CRF_zoom = 3.0
    CRF_zoom_th = 20000

    dataset = 'ss'
    if dataset == 'city':
        print("CITYSCAPES!!!")
    # if CRF_normalize:
    #     mask = (mask - mask.min()) / mask.max()
    #
    # if upscale != 1.0:
    #     mask = torch.clamp(mask*upscale+0.5, min=0, max=1)

    box = box.float()

    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)
    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], im_w)
    box[3] = min(box[3], im_h)
    # print(im_w, im_h)
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    # print(torch.max(mask))
    if upscale != 1.0:
        box_size = w*h
        if box_size > 100:
            upscale = 0.15
        # elif box_size > 40*40:
        #     upscale = 0.2
        else:
            upscale = 0.2
        # print(upscale)
        threshold = sorted(mask.flatten(), reverse=True)[int(box_size * upscale)] + 1e-8

        mask = mask / threshold
        mask = torch.clamp(mask, 0, 1)


    # CRF here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if do_CRF and score > 0.0:
        img_single = img_single[:, :, ::-1]
        h_orig, w_orig = img_single.shape[:2]
        im_mask = torch.zeros((im_h, im_w))
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)
        # print(x_1, y_1, box[2], box[3])
        p_box = [int(x_0), int(y_0), int(x_1), int(y_1)]
        im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]  # projection
        im_mask = im_mask.numpy()

        if dataset == 'city':
            img_crop, cut_point_crop, img_resize, p_box_crop, mask_crop, resize_ratio = crop_city_roi(p_box, img_single, mask=im_mask,
                                                                                                  return_ratio=True)
            # img_crop_norm = norm_image(img_crop)
            h_new, w_new = img_resize.shape[:2]

            mask_for_crf = np.concatenate([np.expand_dims(mask_crop, 0), np.expand_dims(1 - mask_crop, 0)],
                                  axis=0)

            mask_for_crf = crf_postprocessor(img_crop, mask_for_crf)[0]
            # mask_for_crf_norm = crf_postprocessor(img_crop_norm, mask_for_crf)[0]
            # cv2.imshow('img_crop', cv2.resize(img_crop, dsize=None, fx=0.5, fy=0.5))
            # cv2.imshow('img_crop_norm', cv2.resize(img_crop_norm, dsize=None, fx=0.5, fy=0.5))
            #
            # cv2.imshow('mask_crop', cv2.resize((mask_crop * 255.0).astype('uint8'), dsize=None, fx=0.5, fy=0.5))
            # cv2.imshow('crf_crop', cv2.resize((mask_for_crf_nonorm * 255.0).astype('uint8'), dsize=None, fx=0.5, fy=0.5))
            # cv2.imshow('crf_crop_norm', cv2.resize((mask_for_crf_norm * 255.0).astype('uint8'), dsize=None, fx=0.5, fy=0.5))

            # mask_for_crf = mask_for_crf_nonorm
            total_mask = np.zeros([h_new, w_new])
            total_mask[cut_point_crop[1]:cut_point_crop[3], cut_point_crop[0]:cut_point_crop[2]] = mask_for_crf
            mask_for_crf = cv2.resize(total_mask, (2048, 1024))
            # cv2.imshow('img', cv2.resize(img_single, dsize=None, fx=0.5, fy=0.5))
            # cv2.imshow('mask', cv2.resize((im_mask * 255.0).astype('uint8'), dsize=None, fx=0.5, fy=0.5))
            # cv2.imshow('crf', cv2.resize((mask_for_crf * 255.0).astype('uint8'), dsize=None, fx=0.5, fy=0.5))
            # cv2.waitKey(0)

        else:
            if CRF_zoom != 1.0 and box_size < CRF_zoom_th:
                cut_point = [int(box[0] * (1 - 1 / CRF_zoom)), int(box[1] * (1 - 1 / CRF_zoom)),
                             int((w_orig + (CRF_zoom - 1) * box[2]) / CRF_zoom), int((h_orig + (CRF_zoom - 1) * box[3]) / CRF_zoom)]
                before_zoom_size = (cut_point[2] - cut_point[0], cut_point[3] - cut_point[1])
                image_zoom = img_single[cut_point[1]:cut_point[3], cut_point[0]:cut_point[2]]
                mask_zoom = im_mask[cut_point[1]:cut_point[3], cut_point[0]:cut_point[2]]
                image_zoom = cv2.resize(image_zoom, (w_orig, h_orig))
                mask_zoom = cv2.resize(mask_zoom, (w_orig, h_orig))

                mask_for_crf = np.concatenate([np.expand_dims(mask_zoom, 0), np.expand_dims(1 - mask_zoom, 0)], axis=0)
                mask_for_crf = crf_postprocessor(image_zoom, mask_for_crf)[0]
                mask_zoom = cv2.resize(mask_for_crf.copy(), before_zoom_size)
                mask_for_crf = np.zeros(im_mask.shape)
                # print(w_orig, h_orig)
                # print(mask_zoom.shape)
                mask_for_crf[cut_point[1]:cut_point[3], cut_point[0]:cut_point[2]] = mask_zoom
                save_ratio = 1.0

            else:
                mask_for_crf = np.concatenate([np.expand_dims(im_mask, 0), np.expand_dims(1 - im_mask, 0)], axis=0)
                mask_for_crf = crf_postprocessor(img_single, mask_for_crf)[0]
                # cv2.imshow('img', img_single)
                # # cv2.imshow('mask', (im_mask * 255.0).astype('uint8'))
                # cv2.imshow('mask_crf_nozoom', (mask_for_crf * 255.0).astype('uint8'))
                #
                # cv2.waitKey(0)
        mask = torch.Tensor(mask_for_crf)
        if thresh >= 0:
            im_mask = mask > thresh
        else:
            # for visualization and debugging, we also
            # allow it to return an unmodified mask
            im_mask = (mask * 255).to(torch.bool)
    else:
        if thresh >= 0:
            mask = mask > thresh
        else:
            # for visualization and debugging, we also
            # allow it to return an unmodified mask
            mask = (mask * 255).to(torch.bool)

        im_mask = torch.zeros((im_h, im_w), dtype=torch.bool)
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
        ]
    return im_mask


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

class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1, do_CRF=False, upscale=1.0, dataset=None):
        self.threshold = threshold
        self.padding = padding
        self.do_CRF = do_CRF
        self.upscale = upscale
        if self.do_CRF:
            self.crf_postprocessor = DenseCRF(
                iter_max=10,
                pos_xy_std=1,
                pos_w=3,
                bi_xy_std=67,
                bi_rgb_std=3,
                bi_w=4,
            )
            self.dataset = dataset
        else:
            self.crf_postprocessor=None
    def forward_single_image(self, masks, boxes, img_id=None):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        # print(masks[0])
        # print(boxes.bbox[0])
        # print(boxes.get_field('scores')[0])
        if not (img_id is None):
            img_single = self.dataset.__getitem__(img_id, raw_image=True)
            img_single = np.array(img_single)
        else:
            img_single=None

        # multi here
        if self.do_CRF:
            n_jobs = 32
            res = joblib.Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=None)(
                [joblib.delayed(paste_mask_in_image)(mask[0], box, im_h, im_w, self.threshold, self.padding, do_CRF=self.do_CRF, crf_postprocessor=self.crf_postprocessor, img_single=img_single, score=score, upscale=self.upscale, img_id=img_id) for mask, box, score in zip(masks, boxes.bbox, boxes.get_field('scores'))],
            )
        else:
            res = [
                paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding, do_CRF=self.do_CRF, crf_postprocessor=self.crf_postprocessor, img_single=img_single, score=score, upscale=self.upscale)
                for mask, box, score in zip(masks, boxes.bbox, boxes.get_field('scores'))
            ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res


    def __call__(self, masks, boxes, img_id=None):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        if not (img_id == None):
            print(img_id)
        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."
        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        # job lib multiprocess here
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box, img_id=img_id)
            results.append(result)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
