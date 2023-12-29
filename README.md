# 基于ResNet的手写算式识别
> screen shot
![image](https://github.com/tansen87/HandwrittenFormulaRecognition/assets/98570790/13bf3de4-c238-4aa4-b6e2-23e6808ee08e)

### 安装依赖
```bash
pip install -r requirements.txt
```
### 下载训练好的模型
* [模型下载](https://wwm.lanzoub.com/ile0T054abvc) (密码:be5q)
* 将模型拷贝到`model`文件夹中
### 运行
```bash
python main.py
```

##### video demo
[bilibili](https://www.bilibili.com/video/BV1Qa411f7dB/?spm_id_from=333.999.0.0&vd_source=5ee5270944c6e7a459e1311330bf455c)
##### tips
* 目前只训练了<mark>0, 1, 2, 3, 4, 5, 6, 7, 8, 9 , +, -, =,*, /, (, )</mark>符号
* 如果你想自己训练，在[kaggle](https://www.kaggle.com/xainano/handwrittenmathsymbols)下载数据集，然后`train/train.py`进行训练

