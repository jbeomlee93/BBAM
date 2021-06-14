import numpy as np
import scipy.stats as st
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from imageio import imread
import os
import cv2
from tools.BBAM.make_annotation.anno_utils import get_groundtruth, get_groundtruth_coco
from pycocotools import coco



def get_mesh(mu, std=2, xmax=10, ymax=10):
    x = np.linspace(-xmax, xmax, 100)
    y = np.linspace(-ymax, ymax, 100)

    xm, ym = mu

    x, y = np.meshgrid(x, y)
    z = (1 / 2 * np.pi * std * std) * np.exp(-((x - xm) ** 2 / (2 * std ** 2) + (y - ym) ** 2 / (2 * std ** 2)))
    z /= z.max()
    return z.T


def mycdict(color):
    cdict = {
        'red': [(0, 1, 1), (1, 0, 0)],
        'blue': [(0, 1, 1), (1, 0, 0)],
        'green': [(0, 1, 1), (1, 0.05, 0.05)],
        'alpha': [(0, 0, 0), (1, 1, 1)],
    }
    cdict[color] = [(0, 1, 1), (1, 0.6, 0.6)]
    return cdict
    


myreddict = {    
    'red': [(0, 1, 1), (1, 0.6, 0.6)],
    'green': [(0, 1, 1), (1, 0, 0)],
    'blue': [(0, 1, 1), (1, 0.05, 0.05)],    
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}

mybluedict = {
    'red': [(0, 1, 1), (1, 0.03, 0.03)],
    'green': [(0, 1, 1), (1, 0.19, 0.19)],
    'blue': [(0, 1, 1), (1, 0.42, 0.42)],    
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}

mygreendict = {
    'red': [(0, 1, 1), (1, 0, 0)],
    'green': [(0, 1, 1), (1, 0.27, 0.27)],
    'blue': [(0, 1, 1), (1, 0.11, 0.11)],    
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}

mypurpledict = {
    'red': [(0, 1, 1), (1, 0.24, 0.24)],
    'green': [(0, 1, 1), (1, 0, 0)],
    'blue': [(0, 1, 1), (1, 0.49, 0.49)],    
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}

myorangedict = {
    'red': [(0, 1, 1), (1, 0.5, 0.5)],
    'green': [(0, 1, 1), (1, 0.25, 0.25)],
    'blue': [(0, 1, 1), (1, 0.02, 0.02)],    
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}

myorangerealdict = {
    'red': [(0, 1, 1), (1, 1, 1)],
    'green': [(0, 1, 1), (1, 0.5, 0.5)],
    'blue': [(0, 1, 1), (1, 0.0, 0)],
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}
mygraydict = {
    'red': [(0, 1, 1), (1, 0.2, 0.2)],
    'green': [(0, 1, 1), (1, 0.2, 0.2)],
    'blue': [(0, 1, 1), (1, 0.2, 0.2)],
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}

mypinkdict = {
    'red': [(0, 1, 1), (1, 1, 1)],
    'green': [(0, 1, 1), (1, 0.1, 0.1)],
    'blue': [(0, 1, 1), (1, 0.6, 0.6)],
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}
myyellowdict = {
    'red': [(0, 1, 1), (1, 0.8, 0.8)],
    'green': [(0, 1, 1), (1, 0.8, 0.8)],
    'blue': [(0, 1, 1), (1, 0, 0)],    
    'alpha': [(0, 0, 0), (0.5, 0.7, 0.7), (1, 1, 1)],
}


def main2():
    myreds = LinearSegmentedColormap('myreds', mycdict('red'))
    myblues = LinearSegmentedColormap('myblues', mycdict('blue'))
    mygreens = LinearSegmentedColormap('mygreens', mycdict('green'))
    d1 = get_mesh((-5, -5))
    d2 = get_mesh((5, -5))
    d3 = get_mesh((5, 5))

    fig, ax = plt.subplots()
    ax.imshow(d1, cmap=myreds)
    ax.imshow(d2, cmap=myblues)
    ax.imshow(d3, cmap=mygreens)
    plt.show()
    
    
def get_masks(image_id, shape, coco_class):
    dpath = f'{root_path}/{image_id}'
    pidxs = sorted(os.listdir(dpath))
    masks = list()
    for pidx in pidxs:
        if not os.path.exists(f'{dpath}/{pidx}/iter_0299_mask.jpg'):
            continue
        m = imread(f'{dpath}/{pidx}/iter_0299_mask.jpg')

        m = cv2.resize(m, shape[::-1])
        m = np.squeeze(m).astype(np.float32)
        if m.sum() != 0:
            m /= m.max()
        masks.append(m)

    return masks

def get_map(image_id, save_dir=None, coco_class=None, dataset='pascal'):
    if dataset == 'pascal':
        img_name = image_id.split('_')[0]+'_'+image_id.split('_')[1]
        image = imread(f'{image_path}/{img_name}.jpg')
    elif dataset == 'coco':
        image = imread(f'{image_path}/{int(image_id):012d}.jpg')

    masks = get_masks(image_id, image.shape[:2], coco_class)
    pred_image = imread(f'{root_path}/{image_id}/box_prediction.jpg')
    if len(image.shape) == 3:
        # fig, ax = plt.subplots()
        fig = plt.figure()
        aximg = fig.add_subplot(1,2,1)
        ax = fig.add_subplot(1,2,2)
        aximg.imshow(pred_image)
        cmaps = [
            LinearSegmentedColormap('myreds', myreddict),
            LinearSegmentedColormap('myblues', mybluedict),
            LinearSegmentedColormap('mygreens', mygreendict),
            LinearSegmentedColormap('mypurples', mypurpledict),
            LinearSegmentedColormap('myoranges', myorangedict),
            LinearSegmentedColormap('myyellows', myyellowdict),
            LinearSegmentedColormap('mygrays', mygraydict),
            LinearSegmentedColormap('myrealorange', myorangerealdict),
            LinearSegmentedColormap('mypink', mypinkdict),

        ]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cmap='gray', alpha=1)
        ax.imshow(np.ones_like(image) * 255, alpha=0.8)
        ax.set_axis_off()
        for m_idx, mask in enumerate(masks):
            ax.imshow(mask, cmap=cmaps[m_idx%len(cmaps)])

        for cmap, mask in zip(cmaps, masks):
            ax.imshow(mask, cmap=cmap)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, '%s,jpg') % image_id)
        plt.show()

dataset = 'pascal'
if dataset == 'pascal':
    root_path = 'Faster_VOC_val'
    image_path = 'Dataset/VOC2012_SEG_AUG/JPEGImages'

if __name__ == "__main__":
    dataset = 'pascal'
    if dataset == 'pascal':
        root_path = 'Faster_VOC_val'
        image_path = 'Dataset/VOC2012_SEG_AUG/JPEGImages'
        image_ids = [s for s in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, s))]
        image_ids = image_ids
        print(image_ids)


    save_dir = root_path+'_color'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for indexx, image_id in enumerate(image_ids):
        print(indexx, image_id)
        if dataset == 'pascal':
            get_map(image_id, save_dir)
