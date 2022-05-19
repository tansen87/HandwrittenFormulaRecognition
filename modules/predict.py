import os
import cv2
import numpy as np
from modules import segmentation
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image

current_path = os.getcwd()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.BackBone = models.resnet18(pretrained=False)
        self.BackBone.fc = nn.Linear(self.BackBone.fc.in_features, 18)

    def forward(self, x):
        x = self.BackBone(x)
        return x

# 初始化模型
model = Net()
# 检查是否有可用的gpu，如果没有则使用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 权重的路径
model_path = current_path + "\\model\\weight.pth"
# 加载训练好的权重
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
# 进入测试模式，不用计算梯度，速度会快一些
model.eval()
# 对图像做处理
data_transform = transforms.Compose([
    # 缩放到224*224
    transforms.Resize((224, 224)),
    # 将图片转换为tensor
    transforms.ToTensor(),
    # 正则化：降低模型复杂度
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
])
dictionaries = [current_path + '\\modules\\dictionaries.txt']

def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for dic in stuff:
        w = dic.strip().split()
        lexicon[w[0]] = int(w[1])
    return lexicon

word_dicts = load_dict(dictionaries[0])
word_dicts_r = [None] * len(word_dicts)
for kk, vv in word_dicts.items():
    word_dicts_r[vv] = kk

# key和value反向
symbol_names = {value: key for key, value in word_dicts.items()}


def image_predict(image):
    results = ''
    seg_img = segmentation.Vertical_Projection(image)
    for single_img in seg_img:
        image = Image.fromarray(np.uint8(single_img)).convert('RGB')
        r_image = data_transform(image)
        r_image = torch.unsqueeze(r_image, dim=0).float()
        output = model(r_image)
        pred = output.argmax(dim=1, keepdim=True)
        result = symbol_names[int(pred)]
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
