# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, ann_file_ignore=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        self.COCODataset_ignore = torchvision.datasets.coco.CocoDetection(root, ann_file_ignore)
        self.is_ignore = (ann_file_ignore is not None)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.COCODataset_ignore.ids = sorted(self.COCODataset_ignore.ids)
        # filter images without detection annotations
        # if remove_images_without_annotations:
        #     ids = []
        #     for img_id in self.ids:
        #         ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #         anno = self.coco.loadAnns(ann_ids)
        #         if has_valid_annotation(anno):
        #             ids.append(img_id)
        #     self.ids = ids
        # print("len", len(self.ids), len(self.COCODataset_ignore.ids))

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx, raw_image=False):

        img, anno = super(COCODataset, self).__getitem__(idx)
        if raw_image:
            return img
        # filter crowd annotations
        # TODO might be better to add an extra field
        if self.is_ignore:
            _, anno_ignore = self.COCODataset_ignore.__getitem__(idx)
            anno_ignore = [obj for obj in anno_ignore if obj["iscrowd"] == 0]
            boxes_ignore = [obj["bbox"] for obj in anno_ignore]
            boxes_ignore = torch.as_tensor(boxes_ignore).reshape(-1, 4)  # guard against no boxes
            target_ignore = BoxList(boxes_ignore, img.size, mode="xywh").convert("xyxy")
            classes_ignore = [obj["category_id"] for obj in anno_ignore]
            classes_ignore = [self.json_category_id_to_contiguous_id[c] for c in classes_ignore]
            classes_ignore = torch.tensor(classes_ignore)
            target_ignore.add_field("labels", classes_ignore)
            if anno_ignore and "segmentation" in anno_ignore[0]:
                masks_ignore = [obj["segmentation"] for obj in anno_ignore]
                masks_ignore = SegmentationMask(masks_ignore, img.size, mode='poly')
                target_ignore.add_field("masks", masks_ignore)
            target_ignore = target_ignore.clip_to_image(remove_empty=True)
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        # print("box is same?", boxes, boxes_ignore)
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        # print(self._transforms)
        if self._transforms is not None:
            if self.is_ignore:
                img, target, target_ignore = self._transforms(img, target, target_ignore)
            else:
                img, target = self._transforms(img, target)
        if self.is_ignore:
            # print(target)
            # print(target_ignore)
            return img, target, idx, target_ignore
        else:
            return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
