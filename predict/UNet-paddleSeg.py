import matplotlib.pyplot as plt
import changeData
import paddle
from paddleseg.transforms import Compose, Resize
from paddleseg.models import UNet

dataset_path = '../dataset/mini'  # data的文件夹
# 完成三个数据的创建
transforms = [Resize([256, 256])]
train_data = changeData.ChangeDataset(dataset_path, 'train', transforms)
test_data = changeData.ChangeDataset(dataset_path, 'test', transforms)
val_data = changeData.ChangeDataset(dataset_path, 'val', transforms)

model_path = '../train/output/best_model/model.pdparams'  # 加载得到的最好参数
model = UNet(2)
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)

for idx, (img, lab) in enumerate(train_data):  # 从test_data来读取数据
    if idx == 1:  # 查看第三个
        m_img = img.reshape((1, 3, 256, 256))
        m_pre = model(m_img)
        s_img = img.reshape((3, 256, 256)).numpy().transpose(1, 2, 0)
        lab_img = lab.reshape((256, 256)).numpy()
        pre_img = paddle.argmax(m_pre[0], axis=1).reshape((256, 256)).numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(s_img.astype('int64'));plt.title('Time 1')
        plt.show()
        plt.imshow(lab_img);plt.title('Label')
        plt.show()
        plt.imshow(pre_img);plt.title('Change Detection')
        plt.show()
        break  # 只看一个结果就够了