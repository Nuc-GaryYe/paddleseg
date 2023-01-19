from paddleseg.models import BiSeNetV2
model = BiSeNetV2(num_classes=2,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)

import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]

from dataset.dateset import ICSeg
train_dataset = ICSeg(
    dataset_root='dataset/test',
    transforms=transforms,
    mode='train'
)

from dataset.dateset import ICSeg
val_dataset = ICSeg(
    dataset_root='dataset/test',
    transforms=transforms,
    mode='val'
)

import paddle
# 设置学习率
base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)

optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)

from paddleseg.models.losses import CrossEntropyLoss, DiceLoss
losses = {}
losses['types'] = [CrossEntropyLoss()] * 5
losses['coef'] = [1] * 5

print(losses)

from paddleseg.core import train
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output',
    iters=1000,
    batch_size=4,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)