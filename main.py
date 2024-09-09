import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from model.myUnet import MyModel
import torch.optim as optim
import argparse

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from dataloader import *
import utils
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset


class OutputLabelDataset(Dataset):
    def __init__(self, outputs, labels):
        self.outputs = outputs
        self.labels = labels

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        # 确保输出的是 torch.Tensor 而不是 numpy.ndarray
        output = self.outputs[idx]
        label = self.labels[idx]
        return output.numpy(), label.numpy()
    

def test_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images, masks = zip(*batch)  # 解包成两个独立的批次
            image1 =  images[0].to(device)
            image2 =  images[1].to(device).float()


            outputs = model(image1.float()).to(device)
            loss = criterion(outputs, image2)
            running_loss += loss.item() * image1.size(0)



    test_loss = running_loss / len(dataloader.dataset)
    print(f'Test Loss: {test_loss:.4f}')


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            images,masks  = zip(*batch)  # 解包成两个独立的批次
            image1 =  images[0].to(device)
            image2 =  images[1].to(device).float()
            optimizer.zero_grad()

            outputs = model(image1.float())

            loss = model.compute_loss(outputs,image2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image1.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test UNet model")
    # parser.add_argument('--image_dir', type=str, required=True, help='Path to the images directory')
    # parser.add_argument('--mask_dir', type=str, required=True, help='Path to the masks directory')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--dataset', type=str, default='InDoor10', help='dataset')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--a1',type=float, help='loss_g')
    parser.add_argument('--b1', type=float, help='loss_aux1')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 解析命令行参数
    args = parser.parse_args()
    #卡死随机种子
    utils.setup_seed(args.seed)

    ###############
    if args.dataset == 'InDoor10':
        dataset = PositionDataset(
                        DataOption.InDoor10, data_type=15
                        )
    elif args.dataset == 'InDoor10_12':
        dataset = PositionDataset(
                        DataOption.InDoor10, data_type=12
                        )
    elif args.dataset == 'InDoor10_17':
        dataset = PositionDataset(
                        DataOption.InDoor10, data_type=17
                        )
    elif args.dataset == 'InDoor20':
        dataset = PositionDataset(
                        DataOption.InDoor20, data_type=15
                        )
    elif args.dataset == 'InDoor20_12':
        dataset = PositionDataset(
                        DataOption.InDoor20, data_type=12
                        )
    elif args.dataset == 'InDoor20_17':
        dataset = PositionDataset(
                        DataOption.InDoor20, data_type=17
                        )
    elif args.dataset == 'InDoor50':
        dataset = PositionDataset(
                        DataOption.InDoor50, data_type=15
                        )
    elif args.dataset == 'InDoor50_12':
        dataset = PositionDataset(
                        DataOption.InDoor50, data_type=12
                        )
    elif args.dataset == 'InDoor50_17':
        dataset = PositionDataset(
                        DataOption.InDoor50, data_type=17
                        )



    # 使用zip将两个数据集组合在一起
    dataset1 = PositionDataset(DataOption.InDoor10, data_type=15)
    dataset2 = PositionDataset(DataOption.InDoor50, data_type=15)

    
#############################
    combined_dataset = list(zip(dataset1, dataset2))
    # 定义比例
    splits = [0.2, 0.2, 0.2, 0.1, 0.3]
    total_size = len(combined_dataset)
    # 计算每个部分的长度
    split_lengths = [int(total_size * split) for split in splits]

    # 确保总长度一致（因为整数除法可能引入误差）
    # 如果误差存在，将误差调整到最后一个分块中
    split_lengths[-1] = total_size - sum(split_lengths[:-1])

    # 使用 random_split 函数进行数据集拆分
    subsets = random_split(combined_dataset, split_lengths)

    # subsets 将包含多个数据集部分
    firstTrainSet, secondTrainSet, secondGenSet,secondValSet, secondTestSet  = subsets
    

    torch.save(firstTrainSet, 'firstTrainSet.pt')
    torch.save(secondTrainSet, 'secondTrainSet.pt')
    torch.save(secondGenSet, 'secondGenSet.pt')
    torch.save(secondValSet, 'secondValSet.pt')
    torch.save(secondTestSet, 'secondTestSet.pt')
    
########################################


    dataloader = DataLoader(firstTrainSet, batch_size=args.batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = MyModel(args.a1,args.b1).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.test:
        # 训练模型
        model = train_model(model, dataloader, criterion, optimizer, args.num_epochs)
        # 保存模型
        torch.save(model.state_dict(), 'unet_model.pth')
    else:
        # 加载模型
        model.load_state_dict(torch.load('unet_model.pth'))
        # 测试模型
        test_model(model, dataloader, criterion)

##################################################

    # 假设 firstTrainSet 已经被定义并加载
    dataloader = DataLoader(secondGenSet, batch_size=args.batch_size, shuffle=True)

    
    model.eval()  # 切换模型到评估模式
    # 存储处理后的输出与标签的配对
    outputs_list = []
    labels_list = []
    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            images, masks = zip(*batch)  # 解包成两个独立的批次
            
            label = masks[0].to(device).float()
            image1 = images[0].to(device).float()
            
            # 通过模型进行前向传播
            outputs = model(image1)[0]

            
             # 将outputs和labels拆分并逐个添加到各自的列表中
            for i in range(outputs.size(0)):
                outputs_list.append(outputs[i].cpu())
                labels_list.append(label[i].cpu())

        # 创建自定义的 Dataset
    new_dataset = OutputLabelDataset(outputs_list, labels_list)

    # 保存 Dataset
    torch.save(new_dataset, 'model_outputs_labels.pt')
    
#############################################################