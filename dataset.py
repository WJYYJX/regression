import torch
import numpy as np
import skimage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)  # reproducible

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]

    # transforms.Resize(128),
    transforms.Pad(12),
    # transforms.Pad(4),

    transforms.RandomCrop(128),

    #transforms.RandomResizedCrop(224, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.Resize((224, 224),antialias=True)
    #transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    # transforms.ColorJitter(0.2,0.2,0.2),
    #transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # transforms.RandomErasing(p=0.2)

    #transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]

])
'''NPY数据格式'''


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = np.load(data, allow_pickle=True)  # 加载npy数据
        self.label = np.load(label, allow_pickle=True)  # 加载npy数据

        self.transforms = transform  # 转为tensor形式

    def __getitem__(self, index):
        hdct = self.data[index,]  # 读取每一个npy的数据

        # hdct =  hdct[:,:,0:3]
        # print(hdct[:,:,0:3])
        # hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
        ldct = self.label[index,]  # 读取每一个npy的数据array

        ldct = ldct.tolist()

        #ldct = ldct.astype(float)
        # ldct = ldct[0]
        # print(ldct[0])

        # print(hdct.shape,ldct)

        # hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
        # ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255 #加poisson噪声
        # hdct=Image.fromarray(np.uint8(hdct)) #转成image的形式
        # ldct=Image.fromarray(np.uint8(ldct)) #转成image的形式
        # print(hdct.shape)
        hdct = self.transforms(hdct)  # 转为tensor形式
        # ldct= self.transforms(ldct)  #转为tensor形式

        #ldct=torch.Tensor(ldct.astype(None))

        ldct = float(ldct)
        ldct = torch.tensor(ldct)


        return hdct, ldct  # 返回数据还有标签

    def __len__(self):
        return self.data.shape[0]



transform1 = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]

    # transforms.Resize(128),
    # transforms.Pad(12),
    # # transforms.Pad(4),
    #
    # transforms.RandomCrop(128),

    #transforms.RandomResizedCrop(224, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.Resize((224, 224),antialias=True)
    #transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    # transforms.ColorJitter(0.2,0.2,0.2),
    #transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # transforms.RandomErasing(p=0.2)

    #transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]

])
'''NPY数据格式'''


class valDataset(Dataset):
    def __init__(self, data, label):
        self.data = np.load(data, allow_pickle=True)  # 加载npy数据
        self.label = np.load(label, allow_pickle=True)  # 加载npy数据

        self.transforms = transform1  # 转为tensor形式

    def __getitem__(self, index):
         hdct = self.data[index,]  # 读取每一个npy的数据

         # hdct =  hdct[:,:,0:3]
         # print(hdct[:,:,0:3])
         # hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
         ldct = self.label[index,]  # 读取每一个npy的数据array

         ldct = ldct.tolist()

            # ldct = ldct.astype(float)
            # ldct = ldct[0]
            # print(ldct[0])

            # print(hdct.shape,ldct)

            # hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
            # ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255 #加poisson噪声
            # hdct=Image.fromarray(np.uint8(hdct)) #转成image的形式
            # ldct=Image.fromarray(np.uint8(ldct)) #转成image的形式
            # print(hdct.shape)
         hdct = self.transforms(hdct)  # 转为tensor形式
         # ldct= self.transforms(ldct)  #转为tensor形式

         # ldct=torch.Tensor(ldct.astype(None))

         ldct = float(ldct)
         ldct = torch.tensor(ldct)

         return hdct, ldct  # 返回数据还有标签

    def __len__(self):
         return self.data.shape[0]


    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    # import dataset as dataset

# dataset1=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/data/x_data_uint8_choose_train.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/data/y_data_uint8_choose_train.npy')
# train_loader= DataLoader(dataset1, batch_size=args.batch_size  , shuffle=True, pin_memory=True)
# dataset2=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/data/x_data_uint8_choose_test.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/data/y_data_uint8_choose_test.npy')
# test_loader= DataLoader(dataset2, batch_size=args.batch_size  , shuffle=True, pin_memory=True)
