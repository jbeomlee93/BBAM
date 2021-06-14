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
from tools.BBAM.BBAM_FRCNN import process
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
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--ckpt",
    default="logs/ckpoints/Faster_RCNN_final_voc.pth",
    type=str,
)
parser.add_argument(
    "--lr_mask",
    help="Modify config options using the command-line",
    default=0.02,
)
parser.add_argument(
    "--max_iter_mask",
    help="Modify config options using the command-line",
    default=300,
)
parser.add_argument(
    "--tv_beta",
    help="Modify config options using the command-line",
    default=3,
)
parser.add_argument(
    "--tv_coeff",
    help="Modify config options using the command-line",
    default=1e-3,
)
parser.add_argument(
    "--noise_method",
    help="constant | blur | blurrandom",
    default="constant",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
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
l1_coeff = 0.07


image_id = '2007_000727'
queries = [
        [3, 1200, 3]]
print(queries)
results = joblib.Parallel(n_jobs=-1)(
        [joblib.delayed(process)(i) for i in queries]
    )
