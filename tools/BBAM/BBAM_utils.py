import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import os
import math
import torch.nn.functional as F

idx_to_class = {0 : 'aeroplane', 1 : 'bicycle', 2 : 'bird', 3 : 'boat', 4 : 'bottle', 5 : 'bus', 6 : 'car', 7 : 'cat',
                8 : 'chair', 9 : 'cow', 10 : 'table', 11 : 'dog', 12 : 'horse', 13 : 'motorbike', 14 : 'person',
                15 : 'plant', 16 : 'sheep', 17 : 'sofa', 18 : 'train', 19 : 'monitor'}


def tv_norm(input, tv_beta, diagonal=False, sum=False):
    # print(input.shape)
    img = input[0, :]
    if sum:
        row_grad = torch.sum(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
        col_grad = torch.sum(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    else:
        row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
        col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    if diagonal:
        diag = 0
        if sum:
            diag += torch.sum(torch.abs((img[:-1, :-1] - img[1:, 1:])).pow(tv_beta))
            diag += torch.sum(torch.abs((img[1:, :-1] - img[:-1, 1:])).pow(tv_beta))
            diag += torch.sum(torch.abs((img[:-1, 1:] - img[1:, :-1])).pow(tv_beta))
            diag += torch.sum(torch.abs((img[1:, 1:] - img[:-1, :-1])).pow(tv_beta))
        else:
            diag += torch.mean(torch.abs((img[:-1, :-1] - img[1:, 1:])).pow(tv_beta))
            diag += torch.mean(torch.abs((img[1:, :-1] - img[:-1, 1:])).pow(tv_beta))
            diag += torch.mean(torch.abs((img[:-1, 1:] - img[1:, :-1])).pow(tv_beta))
            diag += torch.mean(torch.abs((img[1:, 1:] - img[:-1, :-1])).pow(tv_beta))
        return row_grad + col_grad + diag
    return row_grad + col_grad


def numpy_to_torch(img, requires_grad = True, cuda_device=None):

    use_cuda = torch.cuda.is_available()
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.expand_dims(img, axis=1)
        # output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        if cuda_device==None:
            output = output.cuda()
        else:
            output = output.cuda(cuda_device)


    # output = output.repeat(3, 1, 1)
    v = Variable(output, requires_grad = requires_grad)
    # v = v.repeat(3, 1, 1)

    return v
color_dicts = [
    [0.6, 0, 0.05],
    [0.03, 0.19, 0.42],
    [0, 0.27, 0.11],
    [0.24, 0, 0.49],
    [0.5, 0.25, 0.02],
    [1, 0.5, 0],
    [0.2, 0.2, 0.2],
    [1, 0.1, 0.6],
    [0.8, 0.8, 0]
]
def save_pred(image, boxes, save_path, image_id):
    image[0] += 102.9801
    image[1] += 115.9465
    image[2] += 122.7717
    image = image.data.cpu().numpy().transpose(1, 2, 0).astype('uint8')

    for coord_idx, coords in enumerate(boxes):
        image = cv2.UMat(image).get()
        color = color_dicts[coord_idx%len(color_dicts)]
        color = [int(c*255.0) for c in color]
        color = color[::-1]
        image = cv2.rectangle(image, (int(coords[0]), int(coords[1])),
                                   (int(coords[2]), int(coords[3])), color, 5)

    save_name = '%s/%s/box_prediction.jpg' % (save_path, image_id)
    cv2.imwrite(save_name, image)

def save_mask(mask, masked_img=None, proposal=None, original_coord=None, perturbed_coord=None, iteration=None, proposal_idx=None, image_id=None, class_name=None, save_path_root=None, single_p_idx=None):

    if not (masked_img is None):
        masked_img[0] += 102.9801
        masked_img[1] += 115.9465
        masked_img[2] += 122.7717
        masked_img = masked_img.data.cpu().numpy().transpose(1, 2, 0).astype('uint8')
    mask = (255*mask.data.cpu().numpy().transpose(1, 2, 0)).astype('uint8')
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # blue: proposal, green: unturbed, red_ perturbed

    if (proposal is not None) and (original_coord is not None) and (perturbed_coord is None):
        for coord_idx, coords in enumerate([proposal, original_coord]):
            coords = coords.detach().data.cpu().numpy()

            masked_img = cv2.UMat(masked_img).get()
            masked_img = cv2.rectangle(masked_img, (int(coords[0]), int(coords[1])),
                                       (int(coords[2]), int(coords[3])), color[coord_idx], 5)


    if not((proposal is None) or (original_coord is None) or (perturbed_coord is None)):
        for coord_idx, coords in enumerate([proposal, original_coord, perturbed_coord]):
            coords = coords.detach().data.cpu().numpy()

            masked_img = cv2.UMat(masked_img).get()
            masked_img = cv2.rectangle(masked_img, (int(coords[0]), int(coords[1])),
                                       (int(coords[2]), int(coords[3])), color[coord_idx], 5)



    if not (masked_img is None):
        masked_img = cv2.resize(masked_img, None, fx=0.5, fy=0.5)
        mask = cv2.resize(mask, (masked_img.shape[1], masked_img.shape[0]))
    if single_p_idx is None:
        save_path = '%s/%s/pidx_%04d_%s/' % (save_path_root, image_id, proposal_idx, class_name)
    else:
        save_path = '%s/%s/pidx_%04d_%s/' % (save_path_root, image_id, proposal_idx, class_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if single_p_idx is None:
        if not (masked_img is None):
            cv2.imwrite('%s/iter_%04d.jpg' % (save_path, iteration), masked_img)
        cv2.imwrite('%s/iter_%04d_mask.jpg' % (save_path, iteration), mask)
    else:
        if not (masked_img is None):
            cv2.imwrite('%s/pidx_%04d_img.jpg' % (save_path, single_p_idx), masked_img)
        cv2.imwrite('%s/pidx_%04d_mask.jpg' % (save_path, single_p_idx), mask)


def get_max_iou(source, targets):
    # target: multiple boxes

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

        if not(x_right < x_left or y_bottom < y_top):

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
    return maxIoU

def get_single_iou(source, target):
    # target: multiple boxes

    maxIoU = 0


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
        return 0.0

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

    return iou

def selected_positives(ious, pred_classes, displacements, proposal_iter):
    ious, pred_classes, displacements = np.array(ious), np.array(pred_classes), np.array(displacements)
    top_ious = np.argsort(-ious)
    top_displacement = np.argsort(-displacements)


    # include top 30%
    positive_idxs = list(top_ious[:int(proposal_iter * 0.3)])
    for d in top_displacement:
        if ious[d] > 0.8:
            positive_idxs.append(d)

    return positive_idxs[:proposal_iter]

def imsmooth(tensor,
             sigma,
             stride=1,
             padding=0,
             padding_mode='constant',
             padding_value=0):
    "From TorchRay (https://github.com/facebookresearch/TorchRay)"
    assert sigma >= 0
    width = math.ceil(4 * sigma)
    SQRT_TWO_DOUBLE = torch.tensor(math.sqrt(2), dtype=torch.float32)
    SQRT_TWO_SINGLE = SQRT_TWO_DOUBLE.to(torch.float32)
    EPSILON_SINGLE = torch.tensor(1.19209290E-07, dtype=torch.float32)

    filt = (torch.arange(-width,
                         width + 1,
                         dtype=torch.float32,
                         device=tensor.device) /
            (SQRT_TWO_SINGLE * sigma + EPSILON_SINGLE))
    filt = torch.exp(-filt * filt)
    filt /= torch.sum(filt)
    num_channels = tensor.shape[1]
    width = width + padding
    if padding_mode == 'constant' and padding_value == 0:
        other_padding = width
        x = tensor
    else:
        # pad: (before, after) pairs starting from last dimension backward
        x = F.pad(tensor,
                  (width, width, width, width),
                  mode=padding_mode,
                  value=padding_value)
        other_padding = 0
        padding = 0
    x = F.conv2d(x,
                 filt.reshape((1, 1, -1, 1)).expand(num_channels, -1, -1, -1),
                 padding=(other_padding, padding),
                 stride=(stride, 1),
                 groups=num_channels)
    x = F.conv2d(x,
                 filt.reshape((1, 1, 1, -1)).expand(num_channels, -1, -1, -1),
                 padding=(padding, other_padding),
                 stride=(1, stride),
                 groups=num_channels)
    return x


class MaskGenerator:
    r"""Mask generator.
    The class takes as input the mask parameters and returns
    as output a mask.
    Args:
        shape (tuple of int): output shape.
        step (int): parameterization step in pixels.
        sigma (float): kernel size.
        clamp (bool, optional): whether to clamp the mask to [0,1]. Defaults to True.
        pooling_mehtod (str, optional): `'softmax'` (default),  `'sum'`, '`sigmoid`'.
    Attributes:
        shape (tuple): the same as the specified :attr:`shape` parameter.
        shape_in (tuple): spatial size of the parameter tensor.
        shape_out (tuple): spatial size of the output mask including margin.
    """

    def __init__(self, shape, step, sigma, clamp=True, pooling_method='softmax'):
        self.shape = shape
        self.step = step
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp
        self.pooling_method = pooling_method

        assert int(step) == step

        # self.kernel = lambda z: (z < 1).float()
        self.kernel = lambda z: torch.exp(-2 * ((z - .5).clamp(min=0)**2))

        self.margin = self.sigma
        # self.margin = 0
        self.padding = 1 + math.ceil((self.margin + sigma) / step)
        self.radius = 1 + math.ceil(sigma / step)
        self.shape_in = [math.ceil(z / step) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1
            for z in self.shape_in
        ]
        self.shape_up = [self.step * z for z in self.shape_mid]
        self.shape_out = [z - step + 1 for z in self.shape_up]

        self.weight = torch.zeros((
            1,
            (2 * self.radius + 1)**2,
            self.shape_out[0],
            self.shape_out[1]
        ))

        step_inv = [
            torch.tensor(zm, dtype=torch.float32) /
            torch.tensor(zo, dtype=torch.float32)
            for zm, zo in zip(self.shape_mid, self.shape_up)
        ]

        for ky in range(2 * self.radius + 1):
            for kx in range(2 * self.radius + 1):
                uy, ux = torch.meshgrid(
                    torch.arange(self.shape_out[0], dtype=torch.float32),
                    torch.arange(self.shape_out[1], dtype=torch.float32)
                )
                iy = torch.floor(step_inv[0] * uy) + ky - self.padding
                ix = torch.floor(step_inv[1] * ux) + kx - self.padding

                delta = torch.sqrt(
                    (uy - (self.margin + self.step * iy))**2 +
                    (ux - (self.margin + self.step * ix))**2
                )

                k = ky * (2 * self.radius + 1) + kx

                self.weight[0, k] = self.kernel(delta / sigma)

    def generate(self, mask_in):
        r"""Generate a mask.
        The function takes as input a parameter tensor :math:`\bar m` for
        :math:`K` masks, which is a :math:`K\times 1\times H_i\times W_i`
        tensor where `H_i\times W_i` are given by :attr:`shape_in`.
        Args:
            mask_in (:class:`torch.Tensor`): mask parameters.
        Returns:
            tuple: a pair of mask, cropped and full. The cropped mask is a
            :class:`torch.Tensor` with the same spatial shape :attr:`shape`
            as specfied upon creating this object. The second mask is the same,
            but with an additional margin and shape :attr:`shape_out`.
        """
        mask = F.unfold(mask_in,
                        (2 * self.radius + 1,) * 2,
                        padding=(self.padding,) * 2)
        mask = mask.reshape(
            len(mask_in), -1, self.shape_mid[0], self.shape_mid[1])
        mask = F.interpolate(mask, size=self.shape_up, mode='nearest')
        mask = F.pad(mask, (0, -self.step + 1, 0, -self.step + 1))
        mask = self.weight * mask

        if self.pooling_method == 'sigmoid':
            if self.coldness == float('+Inf'):
                mask = (mask.sum(dim=1, keepdim=True) - 5 > 0).float()
            else:
                mask = torch.sigmoid(
                    self.coldness * mask.sum(dim=1, keepdim=True) - 3
                )
        elif self.pooling_method == 'softmax':
            if self.coldness == float('+Inf'):
                mask = mask.max(dim=1, keepdim=True)[0]
            else:
                mask = (
                    mask * F.softmax(self.coldness * mask, dim=1)
                ).sum(dim=1, keepdim=True)

        elif self.pooling_method == 'sum':
            mask = mask.sum(dim=1, keepdim=True)
        else:
            assert False, f"Unknown pooling method {self.pooling_method}"
        m = round(self.margin)
        if self.clamp:
            mask = mask.clamp(min=0, max=1)
        cropped = mask[:, :, m:m + self.shape[0], m:m + self.shape[1]]
        return cropped, mask

    def to(self, dev):
        """Switch to another device.
        Args:
            dev: PyTorch device.
        Returns:
            MaskGenerator: self.
        """
        self.weight = self.weight.to(dev)
        return self

