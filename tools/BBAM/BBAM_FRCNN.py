import argparse
import os
import torch
import tqdm
import cv2
import numpy as np
import joblib
import torch.nn.functional as F
import shutil
from tools.BBAM.BBAM_utils import numpy_to_torch, save_mask, tv_norm, idx_to_class, get_max_iou, get_single_iou, selected_positives, save_pred
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tools.BBAM.data_utils import BBAM_data_loader
import random
import copy
from maskrcnn_benchmark.structures.bounding_box import BoxList
import time
from tools.BBAM.overlay_heatmaps import get_map
import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
idx_to_class = {0 : 'aeroplane', 1 : 'bicycle', 2 : 'bird', 3 : 'boat', 4 : 'bottle', 5 : 'bus', 6 : 'car', 7 : 'cat',
                8 : 'chair', 9 : 'cow', 10 : 'table', 11 : 'dog', 12 : 'horse', 13 : 'motorbike', 14 : 'person',
                15 : 'plant', 16 : 'sheep', 17 : 'sofa', 18 : 'train', 19 : 'monitor'}

# time.sleep(60*60*2)

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--config-file",
    default="configs/pascal_voc/BBAM_VOC_aug_FPN_R101_FasterRCNN_test.yaml",
    type=str,
)
parser.add_argument(
    "--ckpt",
    default="logs/ckpoints/Faster_RCNN_final_voc.pth",
    type=str,
)
parser.add_argument(
    "--lr_mask",
    default=0.02,
)
parser.add_argument(
    "--max_iter_mask",
    default=300,
)
parser.add_argument(
    "--tv_beta",
    default=3,
)
parser.add_argument(
    "--tv_coeff",
    default=1e-3,
)
parser.add_argument(
    "--noise_method",
    help="constant | blur | blurrandom",
    default="constant",
)
parser.add_argument(
    "--img_idx",
    help="0 to 1449",
    default="1",
)
parser.add_argument(
    "--visualize",
    default=True,
)
parser.add_argument(
    "--gpu",
    default=0,
)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
)


args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


output_dir = cfg.OUTPUT_DIR



mode = 'val'  # val | train_aug_cocostyle
data_dir = 'Dataset/VOC2012_SEG_AUG/'

image_root = os.path.join(data_dir, 'JPEGImages')

save_path_root = 'Faster_VOC_val'
if not os.path.exists(save_path_root):
    os.makedirs(save_path_root)
#


def process(query):

    ind_start, ind_end, gpu_device = query
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_device
    model = build_detection_model(cfg, BBAM=True)
    model.to(cfg.MODEL.DEVICE)
    device = torch.device(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    model.eval()
    BBAM_dataset = BBAM_data_loader(data_dir=data_dir, split=mode, cfg=cfg)

    for ind in range(ind_start, ind_end):
        start_time = time.time()
        image, target, image_id = BBAM_dataset.__getitem__(ind)
        print(ind, image_id)

        _, w, h = image.shape
        images = [image.to(device)]

        with torch.no_grad():
            output, return_for_BBAM, selected_idxs = model(images, return_index=True)
        predictions = [o.to(torch.device("cpu")) for o in output]

        predictions = predictions[0]
        scores_pred = predictions.get_field("scores")
        keep_pred = torch.nonzero(scores_pred > 0.7).squeeze(1)
        positive_idxs = []

        for kkkk in keep_pred:
            positive_idxs.append(selected_idxs[kkkk])


        proposals = return_for_BBAM['proposals'][0]
        class_logits = return_for_BBAM['class_logits']
        box_regression = return_for_BBAM['box_regression']
        after_regression_proposal = return_for_BBAM['after_regression_proposal']

        pred_boxes = []

        print("%s objects are detected ! ! ! " % (len(positive_idxs)))
        for obj_idx, positive_idx in enumerate(positive_idxs):
            print("Explaining %s/%s . . . . ." % ((obj_idx+1), len(positive_idxs)))
            selected_c = torch.argmax(class_logits[positive_idx])
            no_perturb_prob = F.softmax(class_logits[positive_idx].unsqueeze(0), -1)[0].detach()[selected_c]

            result_coord = after_regression_proposal[positive_idx][4 * selected_c:4 * (selected_c + 1)]
            target_area = (result_coord[2]-result_coord[0]) * (result_coord[3]-result_coord[1])

            mask_stride = (16 + 48 * np.power(float(target_area) / (w*h), 0.5))
            mask = np.zeros((int(w / mask_stride), int(h / mask_stride)), dtype=np.float32)
            selected_c = torch.argmax(class_logits[positive_idx]).detach().data.cpu().numpy()
            roi_coord = after_regression_proposal[positive_idx][4 * selected_c:4 * (selected_c + 1)] / mask_stride
            roi_coord = [int(r) for r in roi_coord]
            mask[roi_coord[1]:roi_coord[3], roi_coord[0]:roi_coord[2]] = 1
            proposal_coord = [int(r) for r in proposals.bbox[positive_idx]]
            mask[proposal_coord[1]:proposal_coord[3], proposal_coord[0]:proposal_coord[2]] = 1

            mask = numpy_to_torch(mask)
            optimizer = torch.optim.Adam([mask], lr=args.lr_mask)
            fixed_proposals = return_for_BBAM['proposals']
            pred_boxes.append(after_regression_proposal[positive_idx, 4*selected_c:4*(selected_c+1)].detach().data.cpu().numpy())
            for mask_iter in range(args.max_iter_mask):
                masked_image, _, _ = BBAM_dataset.__getitem__(ind, mask=mask, masking_method=args.noise_method)
                masked_output, masked_return_for_BBAM = model(masked_image, fixed_proposals=fixed_proposals)
                before_reg = box_regression[positive_idx][4 * selected_c:4 * (selected_c + 1)]
                after_reg = masked_return_for_BBAM['box_regression'][positive_idx][
                            4 * selected_c:4 * (selected_c + 1)]
                prob = F.softmax(masked_return_for_BBAM['class_logits'][positive_idx].unsqueeze(0), -1)[0]
                l1_coeff = 0.07

                loss = l1_coeff * torch.sum(torch.abs(mask)) + args.tv_coeff * tv_norm(mask, args.tv_beta, diagonal=True, sum=True) +\
                       torch.abs(before_reg.detach()-after_reg).mean() / 2 + torch.abs(no_perturb_prob- prob[selected_c]) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mask.data.clamp_(0, 1)

                if (mask_iter+1) % 100 == 0:
                    save_mask(mask=mask,
                              masked_img=masked_image,
                              # proposal=fixed_proposals[0].bbox[positive_idx],
                              proposal=masked_return_for_BBAM['proposals'][0].bbox[positive_idx],
                              original_coord=after_regression_proposal[positive_idx, 4*selected_c:4*(selected_c+1)],
                              perturbed_coord=masked_return_for_BBAM['after_regression_proposal'][positive_idx, 4*selected_c:4*(selected_c+1)],
                              iteration=mask_iter,
                              proposal_idx=obj_idx,
                              image_id=image_id,
                              class_name=idx_to_class[selected_c - 1],
                              save_path_root=save_path_root)
        print(time.time() - start_time)

        save_pred(image, pred_boxes, save_path=save_path_root, image_id=image_id,)
        if args.visualize:
            get_map(image_id)
if __name__ == "__main__":
    process([int(args.img_idx), int(args.img_idx)+1, args.gpu])
