#!/usr/bin/env python
# coding: utf-8

# # **基于UNet++的遥感建筑变化检测**
# 在[基于Faster R-CNN的高分辨率遥感影像变化检测](https://www.whu-cveo.com/2018/10/18/paper-fasterrcnn4cd/)中看到，有一种利用深度学习做遥感变化检测的方法为将两时相的图像叠起来形成RGBRGB格式的6通道图像，然后使用语义分割二分割的方法进行变化检测的学习。例如这篇文章就是使用的Faster R-CNN进行变化检测。下面利用PaddleSeg和LEVIR建筑变化数据集试着完成基于UNet++的遥感建筑变化检测

# ## 1. 准备工作
# 这里主要包括
# 1. 数据集的解压、生成数据列表以及构建数据集
# 2. 安装PaddleSeg

# ### 1.1 解压数据及安装PaddleSeg
# **①关于数据集**
# > LEVIR-CD包含637个超高分辨率（VHR，0.5m /像素）的Google Earth图像补丁对，大小为1024×1024像素。这些时间跨度为5到14年的比特影像具有显着的土地利用变化，尤其是建筑的增长。LEVIR-CD涵盖各种类型的建筑物，例如别墅，高层公寓，小型车库和大型仓库。完全注释的LEVIR-CD总共包含31,333个单独的变更构建实例。论文：Chen et al.2020
# > ![](https://ai-studio-static-online.cdn.bcebos.com/fb96cc34fbaa4286b87799e2852d6d3cafdbf6ec5f6247f3ab55658ed0c1d35b)
# 
# **②关于PaddleSeg**
# > PaddleSeg是基于飞桨PaddlePaddle开发的端到端图像分割开发套件，涵盖了高精度和轻量级等不同方向的大量高质量分割模型。通过模块化的设计，提供了配置化驱动和API调用等两种应用方式，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用
# >
# > *Github链接：https://github.com/PaddlePaddle/PaddleSeg*
# >
# > *Gitee链接：https://gitee.com/paddlepaddle/PaddleSeg*

# In[1]:


# 数据集解压
get_ipython().system(' mkdir -p datasets')
get_ipython().system(' mkdir -p datasets/train')
get_ipython().system(' mkdir -p datasets/test')
get_ipython().system(' mkdir -p datasets/val')
get_ipython().system(' unzip -q /home/aistudio/data/data53795/train.zip -d datasets/train')
get_ipython().system(' unzip -q /home/aistudio/data/data53795/test.zip -d datasets/test')
get_ipython().system(' unzip -q /home/aistudio/data/data53795/val.zip -d datasets/val')
# 安装paddleseg
get_ipython().system(' pip -q install paddleseg')


# ### 1.2 生成数据列表
# 在LEVIR-CD中已经分为了train、test和val三个部分，每个部分有A、B和label三个文件夹分别保存时段一、时段二的图像和对应的建筑变化标签（三个文件夹中对应文件的文件名都相同，这样就可以使用replace只从A中就获取到B和label中的路径），list的每一行由空格隔开

# In[13]:


# 生成数据列表
import os

def creat_data_list(dataset_path, mode='train'):
    with open(os.path.join(dataset_path, (mode + '_list.txt')), 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'A')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        A_imgs_name.sort()
        for A_img_name in A_imgs_name:
            A_img = os.path.join(A_path, A_img_name)
            B_img = os.path.join(A_path.replace('A', 'B'), A_img_name)
            label_img = os.path.join(A_path.replace('A', 'label'), A_img_name)
            f.write(A_img + ' ' + B_img + ' ' + label_img + '\n')  # 写入list.txt
    print(mode + '_data_list generated')

dataset_path = 'datasets'  # data的文件夹
# 分别创建三个list.txt
creat_data_list(dataset_path, mode='train')
creat_data_list(dataset_path, mode='test')
creat_data_list(dataset_path, mode='val')


# ### 1.3 构建数据集
# 这里主要是自定义数据Dataset，这里是选择继承paadle.io中的Dataset，而没有使用PaddleSeg中的Dataset，有以下几个问题需要注意
# - 在初始化中transforms、num_classes和ignore_index需要，避免后面PaddleSeg在Eval时报错
# - 这里使用transforms，因为通道数不为3通道，直接使用PaddleSeg的不行Compose会报错，需要将to_rgb设置为False
# - 如果使用Normalize，需要将参数乘6，例如`mean=[0.5]*6`
# - label二值图像中的值为0和255，需要变为0和1，否则啥也学不到，而且计算出来的Kappa系数还为负数
# - 注意图像的组织方式——将两时段的图像在通道层concat起来（22-24行代码）

# In[14]:


# 构建数据集
import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
from paddleseg.transforms import Compose, Resize

batch_size = 2  # 批大小

class ChangeDataset(Dataset):
    # 这里的transforms、num_classes和ignore_index需要，避免PaddleSeg在Eval时报错
    def __init__(self, dataset_path, mode, transforms=[], num_classes=2, ignore_index=255):
        list_path = os.path.join(dataset_path, (mode + '_list.txt'))
        self.data_list = self.__get_list(list_path)
        self.data_num = len(self.data_list)
        self.transforms = Compose(transforms, to_rgb=False)  # 一定要设置to_rgb为False，否则这里有6个通道会报错
        self.is_aug = False if len(transforms) == 0 else True
        self.num_classes = num_classes  # 分类数
        self.ignore_index = ignore_index  # 忽视的像素值
    def __getitem__(self, index):
        A_path, B_path, lab_path = self.data_list[index]
        A_img = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
        B_img = cv2.cvtColor(cv2.imread(B_path), cv2.COLOR_BGR2RGB)
        image = np.concatenate((A_img, B_img), axis=-1)  # 将两个时段的数据concat在通道层
        label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        if self.is_aug:
            image, label = self.transforms(im=image, label=label)
            image = paddle.to_tensor(image).astype('float32')
        else:
            image = paddle.to_tensor(image.transpose(2, 0, 1)).astype('float32')
        label = label.clip(max=1)  # 这里把0-255变为0-1，否则啥也学不到，计算出来的Kappa系数还为负数
        label = paddle.to_tensor(label[np.newaxis, :]).astype('int64')
        return image, label
    def __len__(self):
        return self.data_num
    # 这个用于把list.txt读取并转为list
    def __get_list(self, list_path):
        data_list = []
        with open(list_path, 'r') as f:
            data = f.readlines()
            for d in data:
                data_list.append(d.replace('\n', '').split(' '))
        return data_list

dataset_path = 'datasets'
# 完成三个数据的创建
transforms = [Resize([1024, 1024])]
train_data = ChangeDataset(dataset_path, 'train', transforms)
test_data = ChangeDataset(dataset_path, 'test', transforms)
val_data = ChangeDataset(dataset_path, 'val', transforms)


# ## 2. 模型训练

# ### 2.1 搭建模型及训练
# - 模型：UNet++
# - 损失函数：BCELoss
# - 优化器：Adam
# - 学习率变化：CosineAnnealingDecay
# 
# *这里需要注意的就是模型的输入，需要6个通道，如果使用PaddleSeg中UNet等需要进行相应的修改，这里UNet++可以直接使用in_channels，因此这里直接使用的UNet++。关于UNet++，相比于原始的Unet网络，为了避免UNet中的纯跳跃连接在语义上的不相似特征的融合，UNet++通过引入嵌套的和密集的跳跃连接进一步加强了这些连接，目的是减少编码器和解码器之间的语义差距*
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/4edea84e05d84ef581f85cadb3a534f991d7911285fd4e40ba28aa24d0974774)
# 

# In[18]:


import paddle
from paddleseg.models import UNetPlusPlus
from paddleseg.models.losses import BCELoss
from paddleseg.core import train

# 参数、优化器及损失
epochs = 10
batch_size = 2
iters = epochs * 445 // batch_size
base_lr = 2e-4
losses = {}
losses['types'] = [BCELoss()]
losses['coef'] = [1]

model = UNetPlusPlus(in_channels=6, num_classes=2, use_deconv=True)
# paddle.summary(model, (1, 6, 1024, 1024))  # 可查看网络结构
lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters())  # Adam优化器
# 训练
train(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    optimizer=optimizer,
    save_dir='output',
    iters=iters,
    batch_size=batch_size,
    save_interval=int(iters/10),  # 总共保存10个模型
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)


# ### 2.2 训练过程VDL可视化
# 这里训练的轮数较少，得到的结果不是特别好，但是足以说明训练得到的结果在向好发展
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/0cb14644b38c4566ac1b447ec335974ea7512b5596e5408fa4b17e9c5b8eeeb9)

# ## 3. 变化检测
# 从test_data中测试一组数据，查看训练10论时的best_model有多强的检测能力，这里需要注意的就是
# - 数据拆分为[0:3]和[3:6]分别表示两个时段的图像，类型需要转换为int64才能正常显示
# - 加载模型只用加载模型参数，即.pdparams

# In[18]:


import paddle
from paddleseg.models import UNetPlusPlus
import matplotlib.pyplot as plt

model_path = 'output/best_model/model.pdparams'  # 加载得到的最好参数
model = UNetPlusPlus(in_channels=6, num_classes=2, use_deconv=True)
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)

for idx, (img, lab) in enumerate(test_data):  # 从test_data来读取数据
    if idx == 2:  # 查看第三个
        m_img = img.reshape((1, 6, 1024, 1024))
        m_pre = model(m_img)
        s_img = img.reshape((6, 1024, 1024)).numpy().transpose(1, 2, 0)
        # 拆分6通道为两个3通道的不同时段图像
        s_A_img = s_img[:,:,0:3]
        s_B_img = s_img[:,:,3:6]
        lab_img = lab.reshape((1024, 1024)).numpy()
        pre_img = paddle.argmax(m_pre[0], axis=1).reshape((1024, 1024)).numpy()
        plt.figure(figsize=(10, 10))
        plt.subplot(2,2,1);plt.imshow(s_A_img.astype('int64'));plt.title('Time 1')
        plt.subplot(2,2,2);plt.imshow(s_B_img.astype('int64'));plt.title('Time 2')
        plt.subplot(2,2,3);plt.imshow(lab_img);plt.title('Label')
        plt.subplot(2,2,4);plt.imshow(pre_img);plt.title('Change Detection')
        plt.show()
        break  # 只看一个结果就够了


# ## 4. 结语
# - 可以看到训练10轮半小时的结果还是不错的，大概的建筑变化都检测出来了，再调调参数多跑几轮肯定有更好的效果。
# - 除了通道concat的方法，由于这个变化检测也算有时间序列的特点，可能采用LSTM类也能取得好的效果；或者使用传统的方法（如做差、CVA等）处理两张图得到一张图，再送入网络进行学习【使用CVA详情参考[项目](https://aistudio.baidu.com/aistudio/projectdetail/1604872)】，当然哪样好就不知道了。

# # 关于
# 
# | 姓名 | 陈奕州 |
# | -------- | -------- |
# | 学校/专业 | 重庆交通大学测绘科学与技术 |
# | 主页 | https://aistudio.baidu.com/aistudio/personalcenter/thirdview/1945 |
# 
# 希望和大家一起多交流，特别遥感的同学一起搞起来
