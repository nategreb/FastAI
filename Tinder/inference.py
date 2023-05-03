import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage import io

from nets import ComboNet

sys.path.append('ComboLoss/')


# from: https://github.com/lucasxlu/ComboLoss/

class FacialBeautyPredictor:
    """
    Facial Beauty Predictor
    """

    def __init__(self, pretrained_model_path):
        model = ComboNet(num_out=5, backbone_net_name='SEResNeXt50')
        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # model.load_state_dict(torch.load(pretrained_model_path))

        if torch.cuda.device_count() > 1:
            print("We are running on", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.to(device)
        model.eval()

        self.device = device
        self.model = model

    def infer(self, img_file):
        tik = time.time()
        img = io.imread(img_file)
        img = Image.fromarray(img.astype(np.uint8))

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)
        img = img.to(self.device)

        score, cls = self.model(img)
        tok = time.time()

        return {
            'beauty': float(score.to('cpu').detach().item()),
            'elapse': tok - tik
        }

    def infer_arr(self, image_arr):
        tik = time.time()

        if image_arr.shape[2] == 3:
            image = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        elif image_arr.shape[2] == 4:
            image = cv2.cvtColor(image_arr, cv2.COLOR_BGRA2RGB)
        else:
            image = image_arr.copy()

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = preprocess(image)
        image.unsqueeze_(0)
        image = image.to(self.device)

        score, cls = self.model(image)
        tok = time.time()

        return {
            'beauty': float(score.to('cpu').detach().item()),
            'elapse': tok - tik
        }


if __name__ == '__main__':
    fbp = FacialBeautyPredictor(pretrained_model_path='/home/xulu/ModelZoo/ComboNet.pth')
    print(fbp.infer('../test.jpg'))
