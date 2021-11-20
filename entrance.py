#!/usr/bin/python
# encoding: utf-8
from PyQt5 import QtWidgets
from multiprocessing import Process, Manager, freeze_support
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QSlider, QPushButton, QDesktopWidget, QVBoxLayout, \
    QHBoxLayout, QComboBox, QTextBrowser, QTextEdit, QLabel, QDialog, QFileDialog, QLineEdit, QMessageBox
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5 import QtSql
from PyQt5.QtSql import QSqlQuery
from PyQt5.QtGui import QPixmap, QImage

from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp
from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

import dlib
from scripts.align_all_parallel import align_face
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

experiment_type = 'celebs_super_resolution'

EXPERIMENT_DATA_ARGS = {
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "./test256.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "pretrained_models/psp_ffhq_toonify.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}


def entranceFunction(input_image_path):
    print(EXPERIMENT_DATA_ARGS)
    # ֧�ֿ�ѡ���ݼ�
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']
    pprint.pprint(opts)

    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
    original_image = Image.open(input_image_path)
    if opts.label_nc == 0:
        original_image = original_image.convert("RGB")
    else:
        original_image = original_image.convert("L")
    # ��ȡĬ��ͼƬ ת��ͼƬ��ʽ

    original_image.resize((256, 256))

    # ��ͼƬ���ж���
    input_image = run_alignment(input_image_path)
    input_image.resize((256, 256))

    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(original_image)
    latent_mask = None
    with torch.no_grad():
        tic = time.time()
        result_image = run_on_batch(
            transformed_image.unsqueeze(0), net, latent_mask)[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    input_vis_image = log_input_image(transformed_image, opts)
    output_image = tensor2im(result_image)
    res = np.concatenate([np.array(output_image.resize((256, 256)))], axis=1)
    res_image = Image.fromarray(res)
    plt.imshow(res_image)

    print(res_image)
    res_image.save("./test.jpg")
    print("succeed in processing res Image")
    return "./test.jpg"


def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch(inputs, net, latent_mask=None):
    # 待确定，此处返回值是中间产物还是预处理的模糊
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    # ��Qt��װ
    entranceFunction()

