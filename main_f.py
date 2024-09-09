import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import torchvision.models as models
from torch.nn.functional import normalize
from tqdm import tqdm
from dataloader.dataloader import *
from encoder.convmixer import ConvMixer
import utils
import argparse
import os
from itertools import tee
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from lossfunc import loss_select
from torchsummary import summary
from torchprofile import profile_macs
from thop import profile
from torchstat import stat
from others import CustomDataset



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example program with argparse")
    # 定义命令行参数
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=2021, help='seed')
    parser.add_argument('--model', type=str, default='mpri', help='model')
    # parser.add_argument('--model', type=str, default='frp', help='model')
    parser.add_argument('--every_test', type=bool, default=False, help='every_test')
    parser.add_argument('--print', type=bool, default=True, help='print')
    parser.add_argument('--dataset', type=str, default='InDoor10', help='dataset')
    ###########
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--result_name', type=str, default='ttt', help='result_name')
    parser.add_argument('--loss', type=str, default='e_dis', help='loss')
    parser.add_argument('--other', type=str, default='mse实验0.6', help='mse实验0.6')
    parser.add_argument('--a1',type=float, help='loss_g')
    parser.add_argument('--b1', type=float, help='loss_aux1')
    # 解析命令行参数
    args = parser.parse_args()
    #卡死随机种子
    utils.setup_seed(args.seed)

    ###############

    train_dataset = torch.load('./generate/secondTrainSet.pt')
    val_dataset = torch.load('./generate/secondValSet.pt')
    test_dataset = torch.load('./generate/secondTestSet.pt')   
    gen_dataset = torch.load('./generate/model_outputs_labels.pt')   
    # 将数据包装成Dataset
    gen_dataset = list(zip(gen_dataset, gen_dataset))

    from torch.utils.data import ConcatDataset, DataLoader
    import itertools
    from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
    # 假设 test_dataset 和 gen_dataset 都是 Dataset 对象
    combined_dataset = ConcatDataset([train_dataset, gen_dataset])
    # 加载模型
    model = utils.position_model_select(args.model).to(args.gpu)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=True)
    

    # 创建一个新的DataLoader
    train_loader2 = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

    #选定损失函数
    loss = loss_select(args.loss)
    #
    best_valid_loss = float('inf')
    best_model_state = None
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # 这里的 lr=0.001 是示例值，您可以根据需要调整
    test_losses = []
    #######################
    # 训练循环
    for epoch in tqdm(range(args.epochs)):  # 总训练轮数
        model.train()  # 设置模型为训练模式
        for batch in train_loader2:
            # with open ('train.csv','a+') as file:
            #     file.write(str(label))
            images,masks  = zip(*batch)  # 解包成两个独立的批次
            x =  images[1].to(args.gpu).float()
            label =  masks[1].to(args.gpu).float()
            out = model(x).to(args.gpu)
            print(out.shape)
            L = loss(out.float(), label.float())  # 计算损失

            optimizer.zero_grad()
            L.backward()  # 反向传播
            optimizer.step()  # 更新参数
        ###################
        # 验证阶段
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            total_valid_loss = 0
            for batch in valid_loader:
                images,masks  = zip(*batch)  # 解包成两个独立的批次
                x =  images[1].to(args.gpu).float()
                label =  masks[1].to(args.gpu).float()
                out = model(x.float())
                L = loss(out, label)
                total_valid_loss += L.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)
            print(f'Epoch {epoch}: 验证集损失 {avg_valid_loss}')

            # 保存最佳模型
            if avg_valid_loss < best_valid_loss:
                print('模型更新')
                best_valid_loss = avg_valid_loss
                best_model_state = model.state_dict()
                
                save_path = f'./checkpoint/{args.dataset}_{args.model}_best_model_{args.loss}.pth'
                # 保存模型到磁盘
                torch.save(best_model_state, save_path)

        ################
        # 测试阶段(可以删除)
        if (args.every_test):
            tmp_model = copy.deepcopy(model)#不保存的话训练逻辑就变了,但经过实验发现都一样
            tmp_model.eval()  # 设置模型为评估模式
            # tmp_model.load_state_dict(torch.load(save_path))  # 加载最佳模型,但这样后面都不变了
            with torch.no_grad():  # 关闭梯度计算
                total_test_loss = 0
                for batch in valid_loader:
                    images,masks  = zip(*batch)  # 解包成两个独立的批次
                    x =  images[1].to(args.gpu).float()
                    label =  masks[1].to(args.gpu).float()
                    out = tmp_model(x.float())
                    # 打印损失
                    # L = torch.nn.L1Loss()(out, label)
                    L = utils.cal_average(out, label)
                    total_test_loss += L.item()

                avg_test_loss = total_test_loss / len(test_loader)
                test_losses.append(avg_test_loss)
                print(f'测试集损失: {avg_test_loss}')
    #################
    #创建一个名为 "print" 的文件夹
    if not os.path.exists(f"print/{args.dataset}_{args.model}_{args.loss}"):
        os.makedirs(f"print/{args.dataset}_{args.model}_{args.loss}")
    with open(f"print/{args.dataset}_{args.model}_{args.loss}/test_losses.txt", "a+") as out_loss_file:
            out_loss_file.write(str(test_losses)+'\n')
    #################

    model.load_state_dict(torch.load(save_path))  # 加载最佳模型
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        total_test_loss = 0
        total_test_mae_loss = 0
        total_test_mse_loss = 0
        total_test_rmse_loss = 0
        total_test_mape_loss = 0
        labels = []
        outs = []
        for batch in test_loader:
            images,masks  = zip(*batch)  # 解包成两个独立的批次
            x =  images[1].to(args.gpu).float()
            label =  masks[1].to(args.gpu).float()
            out = model(x.float())
            # 打印损失
        if (args.print):
            labels.append(label.cpu().numpy())
            outs.append(out.cpu().numpy())
        else:
            labels.append(label)
            outs.append(out)

        if (args.print):
            # 合并 labels 和 out 列表中的所有数组
            label = np.concatenate(labels, axis=0)
            out = np.concatenate(outs, axis=0)
            # 将数组转换为字符串，格式化为您想要的样式
            def array_to_formatted_string(arr, var_name):
                # 将数组转换为字符串列表
                str_rows = [", ".join(f"{val:.4f}" for val in row) for row in arr]
                # 将所有行合并为一个大字符串，格式为您所需的格式
                formatted_str = f"{var_name} = [{', '.join('[' + row + ']' for row in str_rows)}]"
                return formatted_str

            # 使用函数格式化 label 和 out
            formatted_label = array_to_formatted_string(label, "Label")
            formatted_out = array_to_formatted_string(out, "Out")

            # 保存到文件
            model_name = args.model  # 假设 args.model 是您的模型名称
            with open(f"print/{args.dataset}_{args.model}_{args.loss}/label.py", "w+") as label_file:
                label_file.write(formatted_label + "\n")

            with open(f"print/{args.dataset}_{args.model}_{args.loss}/out.py", "w+") as out_file:
                out_file.write(formatted_out + "\n")
        

        # Assuming outs and labels are lists
        if (args.print):
            # 将NumPy数组转换为PyTorch张量
            outs = [torch.tensor(o, dtype=torch.float32) for o in outs]
            labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
        else:
            labels.append(label)
            outs.append(out)

        # 在第0个维度合并张量列表
    outs = torch.cat(outs, dim=0)
    labels = torch.cat(labels, dim=0)

    avg_test_loss = utils.cal_average(outs, labels)
    avg_test_mae_loss = torch.nn.L1Loss()(outs, labels)
    avg_test_mse_loss = torch.nn.MSELoss()(outs, labels)
    avg_test_rmse_loss = torch.sqrt(torch.nn.MSELoss()(outs, labels))
    avg_test_mape_loss = torch.mean(torch.abs((labels - outs) / labels))

    print(f'测试集最终评估Dis: {avg_test_loss}')  
    print(f'测试集最终评估MAE: {avg_test_mae_loss}')  
    print(f'测试集最终评估MSE: {avg_test_mse_loss}')  
    print(f'测试集最终评估RMSE: {avg_test_rmse_loss}')  
    print(f'测试集最终评估MAPE: {avg_test_mape_loss}')  


    # 检查文件是否存在
    if not os.path.exists("{file_name}_1050U.csv".format(file_name = args.result_name)):
        # 如果文件不存在，创建文件并写入标题头
        with open("{file_name}_1050U.csv".format(file_name = args.result_name), 'w') as file:
            file.write("a1,b1,dataset,model_name,dis,mae,mse,rmse,mape,seed,loss,other\n")

    #记录结果
    with open("{file_name}_1050U.csv".format(file_name = args.result_name),'a+') as file:
        file.write(str(args.a1) + ',' + str(args.b1) + ',' + args.dataset + ',' + args.model + ',' + str(avg_test_loss)+ ',' + str(avg_test_mae_loss) + ',' + str(avg_test_mse_loss) + ','+ str(avg_test_rmse_loss) + ','+ str(avg_test_mape_loss) + ',' + str(args.seed)+ ','+ str(args.loss) + ','+ str(args.other)+'\n')