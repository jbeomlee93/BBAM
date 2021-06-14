import argparse
import os
import torch
import tqdm
import cv2
import numpy as np
import joblib
import torch.nn.functional as F
import shutil
from tools.BBAM.BBAM_utils import numpy_to_torch, save_mask, tv_norm, idx_to_class, get_max_iou, get_single_iou, selected_positives
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
idx_to_class = {0 : 'aeroplane', 1 : 'bicycle', 2 : 'bird', 3 : 'boat', 4 : 'bottle', 5 : 'bus', 6 : 'car', 7 : 'cat',
                8 : 'chair', 9 : 'cow', 10 : 'table', 11 : 'dog', 12 : 'horse', 13 : 'motorbike', 14 : 'person',
                15 : 'plant', 16 : 'sheep', 17 : 'sofa', 18 : 'train', 19 : 'monitor'}

# time.sleep(60*60*2)

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--config-file",
    default="configs/pascal_voc/BBAM_VOC_aug_FPN_R101_FasterRCNN_test.yaml",
    # metavar="FILE",
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
    default=1e-4,
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


mode = 'train_aug_cocostyle'  # val | train_aug_cocostyle
data_dir = 'Dataset/VOC2012_SEG_AUG/'

image_root = os.path.join(data_dir, 'JPEGImages')
l1_coeff = 0.007

save_path_root = 'BBAM_training_images'
if not os.path.exists(save_path_root):
    os.makedirs(save_path_root)

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
        n_objects = len(target.get_field('labels'))
        proposal_chunks, displacements, target_areas = [], [], []

        for target_coord, target_label in zip(target.bbox, target.get_field('labels')):
            target_coord = target_coord.data.numpy()
            x_len, y_len = target_coord[2]-target_coord[0], target_coord[3]-target_coord[1]
            target_areas.append(x_len*y_len)
            coordinates = []
            d_single = []
            for i in range(1000):
                d_coeff = [random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)]

                displacement_temp = [x_len*d_coeff[0], y_len*d_coeff[1],
                                 x_len*d_coeff[2], y_len*d_coeff[3]]
                perturb_x = [max(target_coord[0]+displacement_temp[0], 0), min(target_coord[2]+displacement_temp[2], h)]
                perturb_y = [max(target_coord[1]+displacement_temp[1], 0), min(target_coord[3]+displacement_temp[3], w)]
                perturb_coord = [perturb_x[0], perturb_y[0], perturb_x[1], perturb_y[1]]
                coordinates.append(perturb_coord)
                d_single.append(np.sum(np.abs(np.array(d_coeff))))
            proposal_chunks.append([BoxList(coordinates, (h, w)).to(device)])
            displacements.append(d_single)
            proposal_chunks[-1][0].add_field("objectness", torch.ones(1000, device=device))


        for obj_idx in range(n_objects):
            check_save_path = '%s/%s/pidx_%04d_%s/iter_0299_mask.jpg' % (save_path_root, image_id, obj_idx, idx_to_class[target.get_field('labels')[obj_idx].numpy() - 1])

            if os.path.exists(check_save_path):
                print(check_save_path, "passed")
                continue

            with torch.no_grad():
                output, return_for_BBAM = model(images, fixed_proposals=proposal_chunks[obj_idx])
            target_box, target_label = target.bbox[obj_idx], target.get_field('labels')[obj_idx]
            class_logits = return_for_BBAM['class_logits']  # 1000*21
            box_regression = return_for_BBAM['box_regression']  # 1000*84
            after_regression_proposal = return_for_BBAM['after_regression_proposal']  # 1000*84
            for iou_th in [0.9, 0.85, 0.8]:
                positive_idxs = []
                for i in range(1000):
                    selected_c = torch.argmax(class_logits[i]).data.cpu().numpy()
                    iou = get_single_iou(target_box, after_regression_proposal[i][4 * selected_c:4 * (selected_c + 1)])
                    if iou > iou_th and selected_c == target_label.numpy():
                        positive_idxs.append(i)
                if len(positive_idxs) > 100:
                    positive_idxs = positive_idxs[:100]
                    break
            if len(positive_idxs) == 0:
                mask_stride = (16 + 48 * np.power(target_areas[obj_idx] / (w*h), 0.5))
                mask = np.zeros((int(w / mask_stride), int(h / mask_stride)), dtype=np.float32)
                mask = numpy_to_torch(mask)

                save_mask(mask=mask,
                          # masked_img=masked_image,
                          # proposal=fixed_proposals[0].bbox[positive_idx],
                          # proposal=masked_return_for_BBAM['proposals'][0].bbox[0],
                          # original_coord=after_regression_proposal[positive_idx, 4 * selected_c:4 * (selected_c + 1)],
                          # perturbed_coord=masked_return_for_BBAM['after_regression_proposal'][0,
                          #                 4 * selected_c:4 * (selected_c + 1)],
                          iteration=299,
                          proposal_idx=obj_idx,
                          image_id=image_id,
                          class_name=idx_to_class[target_label.numpy() - 1],
                          save_path_root=save_path_root)

            if len(positive_idxs) != 0:

                mask_stride = (16 + 48 * np.power(target_areas[obj_idx] / (w*h), 0.5))
                mask = np.zeros((int(w / mask_stride), int(h / mask_stride)), dtype=np.float32)
                selected_c = torch.argmax(class_logits[positive_idxs[0]]).detach().data.cpu().numpy()
                for positive_idx in positive_idxs:
                    roi_coord = after_regression_proposal[positive_idx][4 * selected_c:4 * (selected_c + 1)] / mask_stride
                    roi_coord = [int(r) for r in roi_coord]
                    mask[roi_coord[1]:roi_coord[3], roi_coord[0]:roi_coord[2]] = 1

                mask = numpy_to_torch(mask)
                optimizer = torch.optim.Adam([mask], lr=args.lr_mask)
                no_perturb_cls_probs = F.softmax(class_logits).detach()[:, selected_c]
                fixed_proposal_coords = [proposal_chunks[obj_idx][0].bbox[p_idx].detach().data.cpu().numpy() for p_idx in positive_idxs]
                fixed_proposal = [BoxList(list(fixed_proposal_coords), (h, w)).to(device)]

                for mask_iter in range(args.max_iter_mask):

                    masked_image, _, _ = BBAM_dataset.__getitem__(ind, mask=mask, masking_method=args.noise_method)
                    masked_output, masked_return_for_BBAM = model(masked_image, fixed_proposals=fixed_proposal)
                    loss = 0
                    probs = F.softmax(masked_return_for_BBAM['class_logits'])  # n_proposal * 21
                    for idx, p_idx in enumerate(positive_idxs):
                        before_reg = box_regression[p_idx][4 * selected_c:4 * (selected_c + 1)]
                        after_reg = masked_return_for_BBAM['box_regression'][idx][
                                    4 * selected_c:4 * (selected_c + 1)]
                        loss += torch.abs(before_reg.detach() - after_reg).mean() / 2
                        loss += torch.abs(no_perturb_cls_probs[p_idx] - probs[idx, selected_c]) / 2

                    n_proposals = len(positive_idxs)
                    loss /= n_proposals

                    loss += l1_coeff * torch.sum(torch.abs(mask)) + args.tv_coeff * tv_norm(mask, args.tv_beta, diagonal=True, sum=True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    mask.data.clamp_(0, 1)
                    if (mask_iter == 0) or ((mask_iter+1) % 100 == 0):
                        save_mask(mask=mask,
                                  masked_img=masked_image,
                                  # proposal=fixed_proposals[0].bbox[positive_idx],
                                  proposal=masked_return_for_BBAM['proposals'][0].bbox[0],
                                  original_coord=after_regression_proposal[positive_idx, 4*selected_c:4*(selected_c+1)],
                                  perturbed_coord=masked_return_for_BBAM['after_regression_proposal'][0, 4*selected_c:4*(selected_c+1)],
                                  iteration=mask_iter,
                                  proposal_idx=obj_idx,
                                  image_id=image_id,
                                  class_name=idx_to_class[selected_c - 1],
                                  save_path_root=save_path_root)
        print(time.time() - start_time)
    print("finished", query)


queries = [
           [0, 450, 4], [900, 1350, 4],[1800, 2250, 4], [450, 900, 4], [1350, 1800, 4],[2250, 2700, 4],
            [2700, 3150, 5],[3600, 4050, 5],[4500, 4950, 5], [3150, 3600, 5],[4050, 4500, 5],[4950, 5400, 5],
    [5400, 5850, 6],[6300, 6750, 6],[7200, 7650, 6],[5850, 6300, 6],[6750, 7200, 6],[7650, 8100, 6],
    [8100, 8550, 7], [9000, 9450, 7], [9900, 10200, 7],[8550, 9000, 7], [9450, 9900, 7], [10200, 10582, 7]
           ]

print(queries)
results = joblib.Parallel(n_jobs=-1)(
        [joblib.delayed(process)(i) for i in queries]
    )
