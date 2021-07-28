# PyTorch-Learned-Image-Compression-with-GMM-and-Attention

[English](README.md) | 简体中文

这个项目是cvpr2020文章 [Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Learned_Image_Compression_With_Discretized_Gaussian_Mixture_Likelihoods_and_Attention_CVPR_2020_paper.pdf) 的**PyTorch**实现.

官方代码由 **Tensorflow**实现 [链接](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention).

## 安装

代码测试环境为Ubuntu16.04LTS, CUDA10.1, PyTorch1.2 and Python 3.7。

环境配置已经写在requirements.txt文件中，可以按如下命令配置您的服务器。

```sh
pip install -r requirements.txt
```

## 压缩

### 数据准备

我们首先需要准备训练数据和验证数据。

与原文是用ImageNet 数据集进行训练不同的是，我们采用从filcker.com获得的数据集去训练模型。

你可以按照[链接](https://github.com/liujiaheng/CompressionData)的方式进行训练数据集的准备。

验证集选用 kodak 数据集，可按照如下命令进行获取。

```sh
bash ./data/download_kodak.sh
```

### 训练

对于高比特点 (1024, 2048, 4096), 通道数设置为256，我们采用如下配置文件：

`'config_1024_256.json', 'config_2048_256.json', 'config_4096_256.json'`

对于低比特点 (128, 256, 512), 通道数设置为192，采用如下配置文件：

`'config_128_192.json', 'config_256_192.json', 'config_512_192.json'`

配置文件位于`./examples/example/`.

以比特点为512为例的训练命令如下：

```python
python train.py --config examples/example/config_512_192.json -n baseline_512 --train flick_path --val kodak_path
```

flick_path 是训练数据的路径。

kodak_path是测试数据的路径。

最终你可以找到你的模型文件、日志文件等在`./checkpoints/baseline_512`路径。

你可以将名称`baseline_512`修改为其他。

对于高比特点的训练流程和上面相同， 修改json配置文件即可。

### 测试

测试代码如下，同样以比特点为512为例。

```python
python train.py --config examples/example/config_512_192.json -n baseline_512 --train flick_path --val kodak_path --pretrain pretrain_model_path --test
```

pretrain_model_path 是预训练模型所在路径。

### 模型表现

![pic1](./pic/pic1.png)

**ours** 是我们模型的结果。

**cvpr2020** 是文章 *Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules* 的结果.

### 预训练模型

| bitrate | 链接                                                         |
| ------- | ------------------------------------------------------------ |
| 128     | [85.29MB](https://bhpan.buaa.edu.cn:443/link/676AB03F9A348D939482AA182026377F) |
| 256     | [85.29MB](https://bhpan.buaa.edu.cn:443/link/10A09C6A5905C5418D769693A645DAEC) |
| 512     | [85.29MB](https://bhpan.buaa.edu.cn:443/link/206B2B941E44EF1F59C7DFB7FBDCB0FD) |
| 1024    | [147.60MB](https://bhpan.buaa.edu.cn:443/link/C40A66A5D9DE29233EC8931ABD4721B8) |
| 2048    | [147.60MB](https://bhpan.buaa.edu.cn:443/link/A88041778C62CF596F39EC94C160F2F6) |
| 4096    | [147.60MB](https://bhpan.buaa.edu.cn:443/link/FB24EC4332C331037A8E274C6EC52892) |

