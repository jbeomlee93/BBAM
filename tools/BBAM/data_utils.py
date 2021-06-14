import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import cv2

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.transforms.build import build_transforms

import json
from pycocotools import coco


class BBAM_data_loader(torch.utils.data.Dataset):
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
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    def __init__(self, data_dir, split, cfg, gpu_device=None):
        self.root = data_dir
        self.image_set = split
        self.cfg = cfg
        self.transforms = build_transforms(cfg, is_train=False)
        # print(self.transforms)
        self.keep_difficult = True
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self.gpu_device = gpu_device

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        cls = BBAM_data_loader.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))


    def __getitem__(self, index, mask=None, masking_method='constant', proposal_coord=torch.Tensor([]), batch_size=None):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if mask is not None:
            masked_img, target = self.transforms(img, target)
            if not (batch_size is None):
                masked_img = masked_img.unsqueeze(0).repeat(batch_size, 1, 1, 1)


            if masking_method == 'blur':
                blurred_img = Image.fromarray(cv2.GaussianBlur(np.array(img), (11, 11), 5).astype('uint8'))
                blurred_img, _ = self.transforms(blurred_img, target)
                masked_img = self.masking(masked_img, mask, masking_method, blurred_img, gpu_device=self.gpu_device, proposal_coord=proposal_coord)
            elif masking_method == 'blurrandom':
                blurred_img = cv2.GaussianBlur(np.array(img), (11, 11), 5)/255.0
                noise = np.random.normal(loc=0, scale=1, size=blurred_img.shape)
                blurred_img = np.clip((blurred_img + noise*0.2),0,1)
                blurred_img = Image.fromarray((blurred_img*255.0).astype('uint8'))
                blurred_img, _ = self.transforms(blurred_img, target)
                masked_img = self.masking(masked_img, mask, masking_method, blurred_img, gpu_device=self.gpu_device)

            else:
                masked_img = self.masking(masked_img, mask, masking_method, proposal_coord=proposal_coord, batch_size=batch_size)

            return masked_img, target, img_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_id

    def __len__(self):
        return len(self.ids)


    def masking(self, img, mask, masking_method, blurred_img=None, gpu_device=None, proposal_coord=torch.Tensor([]), batch_size=None):
        # img: torch tensor, (3, w, h), mask: torch tensor
        if gpu_device == None:
            img = img.cuda()
        else:
            img = img.cuda(gpu_device)

        if masking_method == 'constant':
            pass


        elif masking_method == 'blur' or 'blurrandom':
            if gpu_device == None:
                noise_map = torch.Tensor(blurred_img).cuda()
            else:
                noise_map = torch.Tensor(blurred_img).cuda(gpu_device)

        # print(img.shape, mask.shape, noise_map.shape)
        if gpu_device == None:
            if batch_size is None:
                upsample = torch.nn.UpsamplingBilinear2d(size=(img.shape[1], img.shape[2])).cuda()
            else:
                upsample = torch.nn.UpsamplingBilinear2d(size=(img.shape[2], img.shape[3])).cuda()

        else:
            upsample = torch.nn.UpsamplingBilinear2d(size=(img.shape[1], img.shape[2])).cuda(gpu_device)
        if batch_size is None:
            up_mask = upsample(mask.unsqueeze(0))[0]
        else:
            up_mask = upsample(mask)

        if masking_method == 'constant':
            if batch_size is None:
                return_mask = img * up_mask.repeat(3, 1, 1)
            else:
                return_mask = img * up_mask.repeat(1, 3, 1, 1)

            return return_mask
        if proposal_coord.shape[0] == 0:
            return img * up_mask.repeat(3, 1, 1) + (1 - up_mask.repeat(3, 1, 1)) * noise_map
        else:
            proposal_coord = proposal_coord.data.cpu().numpy()
            perturbed_mask = img * up_mask.repeat(3, 1, 1) + (1 - up_mask.repeat(3, 1, 1)) * noise_map
            R = torch.zeros(perturbed_mask.shape).cuda()
            R[:, int(proposal_coord[1]):int(proposal_coord[3]), int(proposal_coord[0]):int(proposal_coord[2])]=1
            px_R = img * R + (1 - R) * noise_map

            return img * up_mask.repeat(3, 1, 1) + (1 - up_mask.repeat(3, 1, 1)) * px_R

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
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
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return BBAM_data_loader.CLASSES[class_id]





class BBAM_data_loader_coco(torch.utils.data.Dataset):

    def __init__(self, data_dir, split, cfg, gpu_device=None):
        self.root = data_dir
        self.image_set = split
        self.cfg = cfg
        self.transforms = build_transforms(cfg, is_train=False)

        self._annopath = os.path.join(self.root, "annotations", "instances_%s2017.json" % split)
        # self.anno_json = json.load(open(self._annopath))
        self.coco_class = coco.COCO(annotation_file=self._annopath)

        self._imgpath = os.path.join(self.root, '%s2017' % split,"%012d.jpg")

        self.gpu_device = gpu_device

        self.ids = sorted(self.coco_class.imgs.keys())
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.match_class = self.class_matching()
        self.classes = self.get_class_list()

    def __getitem__(self, index, mask=None, masking_method='constant', proposal_coord=torch.Tensor([]), batch_size=None):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if mask is not None:
            # print(mask.shape)
            masked_img, target = self.transforms(img, target)
            if not (batch_size is None):
                masked_img = masked_img.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            if masking_method == 'blur':
                blurred_img = Image.fromarray(cv2.GaussianBlur(np.array(img), (11, 11), 5).astype('uint8'))
                blurred_img, _ = self.transforms(blurred_img, target)
                masked_img = self.masking(masked_img, mask, masking_method, blurred_img, gpu_device=self.gpu_device, proposal_coord=proposal_coord)
            elif masking_method == 'blurrandom':
                blurred_img = cv2.GaussianBlur(np.array(img), (11, 11), 5)/255.0
                noise = np.random.normal(loc=0, scale=1, size=blurred_img.shape)
                blurred_img = np.clip((blurred_img + noise*0.2),0,1)
                blurred_img = Image.fromarray((blurred_img*255.0).astype('uint8'))
                blurred_img, _ = self.transforms(blurred_img, target)
                masked_img = self.masking(masked_img, mask, masking_method, blurred_img, gpu_device=self.gpu_device)

            else:
                masked_img = self.masking(masked_img, mask, masking_method, proposal_coord=proposal_coord, batch_size=batch_size)

            return masked_img, target, img_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_id

    def __len__(self):
        return len(self.ids)


    def masking(self, img, mask, masking_method, blurred_img=None, gpu_device=None, proposal_coord=torch.Tensor([]), batch_size=None):
        # img: torch tensor, (3, w, h), mask: torch tensor
        if gpu_device == None:
            img = img.cuda()
        else:
            img = img.cuda(gpu_device)

        if masking_method == 'constant':
            pass

        elif masking_method == 'blur' or 'blurrandom':
            if gpu_device == None:
                noise_map = torch.Tensor(blurred_img).cuda()
            else:
                noise_map = torch.Tensor(blurred_img).cuda(gpu_device)

        # print(img.shape, mask.shape, noise_map.shape)
        if gpu_device == None:
            if batch_size is None:
                upsample = torch.nn.UpsamplingBilinear2d(size=(img.shape[1], img.shape[2])).cuda()
            else:
                upsample = torch.nn.UpsamplingBilinear2d(size=(img.shape[2], img.shape[3])).cuda()

        else:
            upsample = torch.nn.UpsamplingBilinear2d(size=(img.shape[1], img.shape[2])).cuda(gpu_device)
        if batch_size is None:
            up_mask = upsample(mask.unsqueeze(0))[0]
        else:
            up_mask = upsample(mask)

        if masking_method == 'constant':
            if batch_size is None:
                return_mask = img * up_mask.repeat(3, 1, 1)
            else:
                return_mask = img * up_mask.repeat(1, 3, 1, 1)

            return return_mask
        if proposal_coord.shape[0] == 0:
            return img * up_mask.repeat(3, 1, 1) + (1 - up_mask.repeat(3, 1, 1)) * noise_map
        else:
            proposal_coord = proposal_coord.data.cpu().numpy()
            perturbed_mask = img * up_mask.repeat(3, 1, 1) + (1 - up_mask.repeat(3, 1, 1)) * noise_map
            R = torch.zeros(perturbed_mask.shape).cuda()
            R[:, int(proposal_coord[1]):int(proposal_coord[3]), int(proposal_coord[0]):int(proposal_coord[2])]=1
            px_R = img * R + (1 - R) * noise_map

            return img * up_mask.repeat(3, 1, 1) + (1 - up_mask.repeat(3, 1, 1)) * px_R

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno_ids = self.coco_class.getAnnIds(imgIds=img_id)
        anno = self.coco_class.loadAnns(anno_ids)
        anno = self._preprocess_annotation(anno, img_id)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return target

    def _preprocess_annotation(self, target, img_id):
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
            gt_classes.append(self.match_class[obj['category_id']])
        im_info = self.coco_class.imgs[img_id]
        im_info = tuple(map(int, (im_info['height'], im_info['width'])))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    def class_matching(self):
        cats_original = self.coco_class.cats
        norm_cats = {0:0}
        idx = 1
        for cat in cats_original:
            norm_cats[cat] = idx
            idx += 1

        return norm_cats

    def get_class_list(self):
        CATEGORIES = [
            "__background",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        return CATEGORIES

    def idx_to_class(self):
        return dict(zip(range(len(self.classes)), self.classes))
