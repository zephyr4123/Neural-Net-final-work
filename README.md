简要说明，WGAN为项目根目录，下有三个py文件，一个存放图像的文件夹，一个存放数据集的文件夹

## 1. 实验环境
- **操作系统**: Windows 10
- **Python 版本**: 3.11
- **conda 版本**: 4.5.4
- **主要依赖库**:
  - Matplotlib (3.9.2)
  - PyTorch (2.3.1)
  - torchvision (0.18.1a0)

## 2. 数据集下载
- **数据集名称**: MNIST
- **下载链接**: [MNIST 数据集官网](http://yann.lecun.com/exdb/mnist/)
- **数据说明**:
  - 样本数量: 70,000 张图片
  - 样本格式: 灰度图片，28x28 像素
  - 数据来源: Yann LeCun 提供的手写数字图片数据集

## 3. 运行方式
### 3.1 环境配置
1. 安装依赖+clone：
   在conda环境中执行
   conda install matplotlib pytorch torchvision
   克隆项目
   gic clone https://github.com/zephyr4123/Neural-Net-final-work

3. 设置路径：
   数据集下载路径可在源代码如下位置调整。存放在根目录下的data文件夹下，按需调整
   datasets.MNIST(root='./data', train=True, transform=transform, download=True) 
   生成图像保存路径可在源代码如下位置调整。存放在根目录下的generated_images文件夹下，按需调整
   output_dir = "./generated_images"

### 3.2 运行命令
三个py文件对应WGAN、WGAN-GP、谱归一化WGAN，直接运行即可训练，并在训练结束后直接输出损失曲线并将生成图像存入generated_imades下（覆盖）
运行主程序：
python wgan_train.py --WGAN
python wgan_train_penalty.py  --WGAN-GP
python wgan_train_normalize.py  --谱归一化WGAN


可选参数说明：
- `--epochs`: 训练的轮数，默认值为 20
- `--batch_size`: 批量大小，默认值为 512
- `--latent_dim`: 隐空间大小，默认值为 100
- `--image_dim`: 图像大小，默认值为 28*28
- `--lr`: 学习率，默认值为 1e-4
- `--critic_iterations`: 判定器迭代次数，默认值为 10
- `--clip_value`: 权重裁剪范围，默认值为 0.01
- `--lambda_gp`: 梯度惩罚系数，默认值为 10

## 4. 实验结果
因该课题为图像生成，结果均为图片，结果较多在此不做展示，具体结果与分析请看报告

---

## 附录
- **代码仓库**: [GitHub 仓库地址](https://github.com/zephyr4123/Neural-Net-final-work)
- **数据来源**: [MNIST 数据集官网](http://yann.lecun.com/exdb/mnist/)

