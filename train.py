import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import npydatasete as dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from tensorboardX import SummaryWriter

# from model import resnet101,resnet50,resnet18
from trans_res import *
from xvdaoliang import *

import pdb

import regressionhead as regressionhead
#from torchsummary import summary

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    tb_writer = SummaryWriter()
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = dataset.MyDataset('/home/wjy/numpydata/goubang_1.2.4.6.10.16/train/image', '/home/wjy/numpydata/goubang_1.2.4.6.10.16/train/label')#'./x_train_mix.npy', './y_train_mix.npy'  './train_data.npy', './train_label.npy' './x_train_total.npy', './y_train_total.npy'
    train_num = len(train_dataset)

    #/home/wjy/numpydata/jinzhounew1_1.2.4.6.10.16/train/image
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    #flower_list = train_dataset.class_to_idx
    #cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    #json_str = json.dumps(cla_dict, indent=4)
   # with open('class_indices.json', 'w') as json_file:
   #    json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,collate_fn=train_dataset.collate_fn)

    validate_dataset = dataset.valMyDataset('/home/wjy/numpydata/xinjiang_select_1.2.4.6.10.16_copy1/val/image', '/home/wjy/numpydata/xinjiang_select_1.2.4.6.10.16_copy1/val/label')#'./x_val_mix.npy', './y_val_mix.npy'  './val_data.npy', './val_label.npy' './x_val_total.npy', './y_val_total.npy'
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,collate_fn=train_dataset.collate_fn)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net = resnet50(num_classes=1, include_top=True)
    # net = resnet101(include_top=False)

    # net = resnet18(num_classes=1, include_top=True)
    # net = regressionhead.head(net)

    backbone = models.resnet18(pretrained=False)
    net = ResNet18(backbone, num_classes=1)

#     net = MultiChannelLSTM(
#     in_channels=6,
#     base_channels=16,
#     feature_dim=256,
#     lstm_hidden_size=512
# )

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "/home/wjy/regression1/pytorch_classification/Test5_resnet/resnet50_junyunmoniyan_new_pretrain.pth"
    # print(model_weight_path)
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False
    # for name, param in net.named_parameters():
    #     if 'mlp_head' in name:
    #         param.requires_grad = True
    # # # for p in net.backbone.layer1.parameters(): p.requires_grad = False
    # # # for p in net.backbone.layer2.parameters(): p.requires_grad = False
    # for p in net.backbone.layer3.parameters(): p.requires_grad = True
    # for p in net.backbone.layer4.parameters(): p.requires_grad = True



    # print(net)
    # change fc layer structure
    #in_channel = net.fc.in_features
    #net.fc = nn.Linear(in_channel, 1)

    net.to(device)
    #print(net)
    #summary(net, (6, 224, 224))
    # define loss function
    loss_function = nn.L1Loss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")



    optimizer = optim.Adam(params, lr=0.00001)

    # optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.003)

    # def lr_lambda(epoch):
    #     if epoch < 120:
    #         return 1.0
    #     elif epoch < 240:
    #         return 0.5
    #     else:
    #         return 0.1
    #
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    epochs = 300
    best_acc = 0.0
    best_acc1 = 0
    best_mse = 10000
    best_mse1 = 10000
    save_path = './{}Net.pth'.format('CFGN')
    train_steps = len(train_loader)
    best_train_accurate = 0
    best_train_accurate1 = 0
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        acc_train = 0.0
        acc_train1 = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            outputs = outputs.squeeze(-1)
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # scheduler.step(epoch)
            # print statistics
            running_loss += loss.item()
            current_lr = 0.0005
            # current_lr = scheduler.get_last_lr()[0]

            nums = labels.shape
            nums = nums[0]
            for num in range(nums):
                if (labels[num] - 0.5).to(device) <= outputs[num].to(device) <= (labels[num] + 0.5).to(device):
                    acc_train = acc_train + 1
            train_accurate = acc_train / train_num
            if train_accurate >= best_train_accurate:
                best_train_accurate = train_accurate

            for num in range(nums):
                if (labels[num] - 1).to(device) <= outputs[num].to(device) <= (labels[num] + 1).to(device):
                    acc_train1 = acc_train1 + 1
            train_accurate1 = acc_train1 / train_num
            if train_accurate1 >= best_train_accurate1:
                best_train_accurate1 = train_accurate1

            if loss.item() <= best_mse:
                best_mse = loss.item()
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)
            train_bar.desc = "train epoch[{}/{}] loss: {:.3f}, mse: {:.3f},best-acc0.5: {:.3f}, best-acc1: {:.3f}, 0.5acc:{:.3f} , 1acc:{:.3f} ,lr:{:.4f}".format(
                epoch + 1,
                epochs,
                running_loss / (step + 1),
                best_mse,
                best_train_accurate, best_train_accurate1,
                train_accurate, train_accurate1, current_lr)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        acc1 = 0.0

        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # predict_y = torch.max(outputs, dim=1)[1]
                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                outputs = outputs.squeeze(-1)  # 下降一维度
                val_loss = loss_function(outputs, val_labels.to(device))

                nums_val = val_labels.shape
                nums_val = nums_val[0]
                for num in range(nums_val):
                    if (val_labels[num] - 0.5).to(device) <= outputs[num].to(device) <= (val_labels[num] + 0.5).to(
                            device):
                        acc = acc + 1

                accu_loss += val_loss
                val_accurate = acc / val_num
                for num in range(nums_val):
                    if (val_labels[num] - 1).to(device) <= outputs[num].to(device) <= (val_labels[num] + 1).to(device):
                        acc1 = acc1 + 1
                val_accurate1 = acc1 / val_num

                if val_loss.item() <= best_mse1:
                    best_mse1 = val_loss.item()

                val_bar.desc = "valid epoch[{}/{}] loss: {:.3f},mse: {:.3f}, best-acc0.5: {:.3f}, best-acc1: {:.3f}, 0.5acc:{:.3f}, 1acc:{:.3f}".format(
                    epoch + 1,
                    epochs,
                    accu_loss.item() / (
                            step + 1),
                    best_mse1, best_acc, best_acc1,
                    val_accurate, val_accurate1)

        val_accurate = acc / val_num
        # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #       (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        if val_accurate1 > best_acc1:
            best_acc1 = val_accurate1

        tags = ["train_loss", "train_acc0.5", "val_loss", "val_acc0.5", "learning_rate", "train_acc1", "val_acc1",
                "train_mse", "val_mse"]
        tb_writer.add_scalar(tags[0], running_loss, epoch)
        tb_writer.add_scalar(tags[1], train_accurate, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_accurate, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], train_accurate1, epoch)
        tb_writer.add_scalar(tags[6], val_accurate1, epoch)
        tb_writer.add_scalar(tags[7], loss.item(), epoch)
        tb_writer.add_scalar(tags[8], val_loss.item(), epoch)

    print('Finished Training')


if __name__ == '__main__':
    main()
