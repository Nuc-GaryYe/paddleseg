import changeData
import paddle
from paddleseg.transforms import Compose, Resize
from paddleseg.models import UNet
from paddleseg.models.losses import BCELoss
from paddleseg.core import train

dataset_path = '../dataset/fullData'  # data的文件夹

# 完成三个数据的创建
transforms = [Resize([256, 256])]
train_data = changeData.ChangeDataset(dataset_path, 'train', transforms)
test_data = changeData.ChangeDataset(dataset_path, 'test', transforms)
val_data = changeData.ChangeDataset(dataset_path, 'val', transforms)
batch_size = 1  # 批大小

model = UNet(2)
paddle.summary(model, (1, 3, 256, 256))  # 可查看网络结构

# 参数、优化器及损失
epochs = 20
batch_size = 1
iters = epochs * 20 // batch_size

losses = {}
losses['types'] = [BCELoss()]
losses['coef'] = [1]

base_lr = 2e-4

lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters())  # Adam优化器

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