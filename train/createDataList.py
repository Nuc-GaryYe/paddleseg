import os
import random

#参考遥感变化检测，去掉了B
def create_data_list(dataset_path, train_rate, val_rate):
    trainRange = 1.0 * train_rate
    valRange = 1.0 * (train_rate + val_rate)

    with open(os.path.join(dataset_path, ('train_list.txt')), 'w') as f_train:
        with open(os.path.join(dataset_path, ('val_list.txt')), 'w') as f_val:
            with open(os.path.join(dataset_path, ('test_list.txt')), 'w') as f_test:
                A_path = os.path.join(os.path.join(dataset_path), 'A')
                A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
                A_imgs_name.sort()

                for A_img_name in A_imgs_name:
                    A_img = os.path.join(A_path, A_img_name)
                    path1 = os.path.join(A_path.replace('A', 'label'), A_img_name)
                    label_img = path1.replace('jpg', 'png')

                    randomNum = random.random()
                    if(randomNum < trainRange):
                        f_train.write(A_img + ' ' + label_img + '\n')
                    elif(randomNum < valRange):
                        f_val.write(A_img + ' ' + label_img + '\n')
                    else:
                        f_test.write(A_img + ' ' + label_img + '\n')

    print('data_list generated')

if __name__ == "__main__":
    random.seed(20230206)  # 保证每次随机抽取的都可以复现
    train_rate = 0.6
    val_rate  = 0.2
    test_rate = 0.2

    dataset_path = '../dataset/fullData'  # data的文件夹
    create_data_list(dataset_path, train_rate, val_rate)