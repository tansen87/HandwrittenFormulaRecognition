# 基于ResNet的手写公式识别
> Notice: python版本使用 ^3.9

![image](https://github.com/tansen87/HandwrittenFormulaRecognition/assets/98570790/13bf3de4-c238-4aa4-b6e2-23e6808ee08e)

### 运行项目

1. 安装uv

   ```bash
   pip install uv
   ```

2. 使用uv创建虚拟环境

   ```bash
   uv venv
   ```

3. 激活创建的虚拟环境

   ```bash
   .venv\Scripts\activate
   ```

4. 安装依赖

   ```bash
   uv pip install -r requirements.txt
   ```

5. 运行项目

   ```bash
   uv run main.py
   ```

#### video demo
[bilibili](https://www.bilibili.com/video/BV1Qa411f7dB/?spm_id_from=333.999.0.0&vd_source=5ee5270944c6e7a459e1311330bf455c)

#### tips
* 目前只训练了<mark>0, 1, 2, 3, 4, 5, 6, 7, 8, 9 , +, -, =,*, /, (, )</mark>符号
* 如果你想自己训练，在[kaggle](https://www.kaggle.com/xainano/handwrittenmathsymbols)下载数据集，然后`train/train.py`进行训练
