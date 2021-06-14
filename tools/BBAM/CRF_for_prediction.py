import torch
import os

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
import time



parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument(
    "--config-file",
    default="configs/pascal_voc/BBAM_VOC_aug_FPN_R101_MaskRCNN.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--ckpt",
    help="The path to the checkpoint for test, default is the latest checkpoint.",
    default=None,
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


output_name = 'BBAM_Mask_RCNN_logs_mcg85'
output_folder = '%s/inference/voc_2012_val_cocostyle/' % output_name

extra_args = dict(
    box_only=False,
    iou_types=["bbox", "segm"],
    expected_results=[],
    expected_results_sigma_tol=4,
)
dataset_names = cfg.DATASETS.TEST
data_loaders = make_data_loader(cfg, is_train=False, is_distributed=False)[0]
dataset = data_loaders.dataset
print(len(dataset))
predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
upscale=0.6

output_folder = output_folder + 'after_crf'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
evaluate(dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        do_CRF=True,
        upscale=upscale,
        **extra_args,
        )
# search "# job lib multiprocess here" to implement multi-processing

print(output_folder)
# print("17500")