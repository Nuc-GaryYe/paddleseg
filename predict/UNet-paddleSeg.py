import matplotlib.pyplot as plt
import changeData
import paddle
from paddleseg.transforms import Compose, Resize
from paddleseg.models import UNet
from paddleseg.core import evaluate
from connectAcc import getConnectedAcc
import cv2
import numpy as np

dataset_path = '../dataset/mini'  # data的文件夹
# 完成三个数据的创建
transforms = [Resize([512, 512])]
train_data = changeData.ChangeDataset(dataset_path, 'train', transforms)
test_data = changeData.ChangeDataset(dataset_path, 'test', transforms)
val_data = changeData.ChangeDataset(dataset_path, 'val', transforms)

model_path = '../output/best_unet/model.pdparams'  # 加载得到的最好参数
model = UNet(2)
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)

#评估函数
#evaluate(model, train_data)

import paddle.nn.functional as F
acc = 0
num = 0
for idx, (img, lab) in enumerate(train_data):  # 从test_data来读取数据
    m_img = img.reshape((1, 3, 512, 512))
    m_pre = model(m_img)

    s_img = img.reshape((3, 512, 512)).numpy().transpose(1, 2, 0)

    pre_img = paddle.argmax(m_pre[0], axis=1).reshape((512, 512)).numpy()

    pre_img = (np.maximum(pre_img, 0) / pre_img.max()) * 255.0
    pre_img = np.uint8(pre_img)
    num_objects, labels = cv2.connectedComponents(pre_img, connectivity=4);

    print(num_objects)

    lab_img = lab.reshape((512, 512)).numpy()
    lab_img = (np.maximum(lab_img, 0) / lab_img.max()) * 255.0
    lab_img = np.uint8(lab_img)
    num_object2, labels = cv2.connectedComponents(lab_img, connectivity=4);

    print(num_object2)

    if  num_objects >num_object2:
        acc = acc + num_object2 / num_objects
        print(num_object2 / num_objects)
    else:
        acc = acc + num_objects / num_object2
        print(num_objects / num_object2)
    num = num + 1

print(acc / num)

    # if idx == 3:  # 查看第三个
    #     m_img = img.reshape((1, 3, 512, 512))
    #     m_pre = model(m_img)
    #
    #     #测试loss函数
    #     # m_pre = model(m_img)[0]
    #     # print(m_pre)
    #     # label = lab.squeeze(1)
    #     # label = F.one_hot(label, m_pre.shape[1])
    #     # label = label.transpose((0, 3, 1, 2))
    #     # print(label)
    #     # output = paddle.nn.functional.binary_cross_entropy_with_logits(m_pre, label)
    #     # print(output)
    #
    #     s_img = img.reshape((3, 512, 512)).numpy().transpose(1, 2, 0)
    #     lab_img = lab.reshape((512, 512)).numpy()
    #     pre_img = paddle.argmax(m_pre[0], axis=1).reshape((512, 512)).numpy()
    #     print(pre_img)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(s_img.astype('int64'));plt.title('Time 1')
    #     plt.show()
    #     plt.imshow(lab_img, cmap=plt.cm.gray);plt.title('Label')
    #     plt.show()
    #     plt.imshow(pre_img, cmap=plt.cm.gray);plt.title('Change Detection')
    #     plt.show()
    #     break  # 只看一个结果就够了
