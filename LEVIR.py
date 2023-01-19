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
            f.write(A_img + ' ' + label_img + '\n')  # 写入list.txt
    print(mode + '_data_list generated')

dataset_path = 'dataset/test'  # data的文件夹
# 分别创建三个list.txt
creat_data_list(dataset_path, mode='train')
creat_data_list(dataset_path, mode='test')
creat_data_list(dataset_path, mode='val')

import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
from paddleseg.transforms import Compose, Resize

batch_size = 1  # 批大小
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
        A_path, lab_path = self.data_list[index]
        #A_path, B_path, lab_path = self.data_list[index]
        # A_img = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
        # B_img = cv2.cvtColor(cv2.imread(B_path), cv2.COLOR_BGR2RGB)
        # image = np.concatenate((A_img, B_img), axis=-1)  # 将两个时段的数据concat在通道层
        image = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
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

# 完成三个数据的创建
transforms = [Resize([512, 512])]
train_data = ChangeDataset(dataset_path, 'train', transforms)
test_data = ChangeDataset(dataset_path, 'test', transforms)
val_data = ChangeDataset(dataset_path, 'val', transforms)

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

model = UNetPlusPlus(in_channels=3, num_classes=2, use_deconv=True)
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