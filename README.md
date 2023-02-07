# paddleseg
基于paddleSeg修改
! pip -q install PaddlePaddle==2.0.2
! pip -q install paddleseg==2.4.0

所有模型最后一部返回的都是卷积的结果，没有进行激活函数
BCELOSS是再损失函数中先进行sigmoid激活的

模型返回的是一个tensor的列表，如果只有一个的话取【0】
 