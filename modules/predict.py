import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import ResNet
from torchvision import transforms, models

from modules import segmentation


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.BackBone = models.resnet18(pretrained=False)
        self.BackBone.fc = nn.Linear(self.BackBone.fc.in_features, 18)

    def forward(self, x) -> ResNet:
        x = self.BackBone(x)
        return x


class MyModel:
    def __init__(self) -> None:
        # 初始化模型
        self.model = Net()
        # 检查是否有可用的gpu，如果没有则使用cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 权重的路径
        self.model_path = f"{os.getcwd()}/model/weight.pth"
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        # 进入测试模式，不用计算梯度，速度会快一些
        self.model.eval()
        # 对图像做处理
        self.data_transform = transforms.Compose([
            # 缩放到224*224
            transforms.Resize((224, 224)),
            # 将图片转换为tensor
            transforms.ToTensor(),
            # 正则化：降低模型复杂度
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        ])
        self.dictionaries: list[str] = [f"{os.getcwd()}/modules/dictionaries.txt"]

    def get_symbol(self) -> dict:
        word_dicts = self.load_dict(self.dictionaries[0])
        word_dicts_r = [None] * len(word_dicts)
        for kk, vv in word_dicts.items():
            word_dicts_r[vv] = kk
        # key和value反向
        symbol_names = {value: key for key, value in word_dicts.items()}
        return symbol_names

    def load_dict(self, dictFile: str) -> dict:
        with open(dictFile, "r") as fp:
            stuff = fp.readlines()
        lexicon: dict = {}
        for dic in stuff:
            w = dic.strip().split()
            lexicon[w[0]] = int(w[1])
        return lexicon

    def image_predict(self, image: str) -> str:
        results: str = ''
        seg_img = segmentation.Vertical_Projection(image)
        for single_img in seg_img:
            image = Image.fromarray(np.uint8(single_img)).convert('RGB')
            r_image = self.data_transform(image)
            r_image = torch.unsqueeze(r_image, dim=0).float()
            output = self.model(r_image)
            pred = output.argmax(dim=1, keepdim=True)
            result = self.get_symbol()[int(pred)]
            if result == 'times':
                result = '*'
            elif result == 'div':
                result = '/'
            elif result == '(':
                result = '('
            elif result == ')':
                result = ')'
            elif result == ',':
                result = '.'
            results += result
        return results
