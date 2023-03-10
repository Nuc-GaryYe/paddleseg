import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
from paddleseg.transforms import Compose, Resize

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

if __name__ == "__main__":
    dataset_path = 'dataset/mini'
    transforms = [Resize([512, 512])]
    train_data = ChangeDataset(dataset_path, 'train', transforms)
    test_data = ChangeDataset(dataset_path, 'test', transforms)
    val_data = ChangeDataset(dataset_path, 'val', transforms)
