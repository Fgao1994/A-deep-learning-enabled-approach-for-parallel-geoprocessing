from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
# import torch.nn.functional as F
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage import transform
from sklearn.metrics import r2_score
from data_read import DEMDataset
import os
# os.environ['TORCH_HOME']='F:/Python/ml_env/dl/pretrain_model'
os.environ['TORCH_HOME']='/home/fgao/LoadBalance/PeakPoint/python/ml_env/dl/pretrain_model'

plt.ion()   # interactive mode

class Rescale(object):
    """Rescale the image in a sample to a given size.

 Args:
 output_size (tuple or int): Desired output size. If tuple, output is
 matched to output_size. If int, smaller of image edges is matched
 to output_size keeping aspect ratio the same.
 """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

 Args:
 output_size (tuple or int): Desired output size. If int, square crop
 is made.
 """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image.shape[0]
        image = image.reshape((1, image.shape[0], image.shape[1]))
        labelL = []
        labelL.append(label)
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array(labelL)).float()}

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = (image - self.mean)/self.std
        # image = F.normalize(image, self.mean, self.std, self.inplace)
        return {'image': image, 'label': label}

# 函数：trainning
def train_model(model, criterion, optimizer, scheduler, num_epoches=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    for epoch in range(num_epoches):
        print(f"Epoch {epoch}/{num_epoches}")
        print("-"*10)
        y_predict = []
        y_true = []
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for i, data in enumerate(dataloader[phase]):
                inputs = data['image']
                labels = data['label'] #/ 1000
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    labels_array = labels.detach().numpy().flatten()
                    outputs_array = outputs.detach().numpy().flatten()
                    for ele in labels_array: y_true.append(ele)
                    for ele in outputs_array: y_predict.append(ele)
#                     if i % 100 == 99:
#                         print(f'Loss of batch {i} in epoch {epoch}: {loss.item()}')
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0) #这里乘以inputs.size(0),是为了得到批次里样本的loss和，下面的epoch_loss就是每个样本的误差

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_r2 = r2_score(y_true, y_predict)

            print('{} Loss and r2: {:.4f}, {:.4f}'.format(
                    phase, epoch_loss, epoch_r2))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model

print(torch.__version__)

# test1
# mean = 3126.413
# std =  401.522

# test2
# mean = 3094.995
# std =  401.218


# test3
mean = 2096.526
std =  971.579

# trainset = DEMDataset(r'F:\C++\linux\aigis\peakpoint\model\test4\train.csv',
#                       r'F:\C++\linux\aigis\peakpoint\model\test4\sample_image',
#                       transform=transforms.Compose([
#                           #Rescale(256),
#                           RandomCrop(224),
#                           ToTensor(),
#                           Normalize(mean, std)
#                       ]))
# testset = DEMDataset(r'F:\C++\linux\aigis\peakpoint\model\test4\test.csv',
#                       r'F:\C++\linux\aigis\peakpoint\model\test4\sample_image',
#                      transform=transforms.Compose([
#                          #Rescale(256),
#                          RandomCrop(224),
#                          ToTensor(),
#                         Normalize(mean, std)
#                      ]))

trainset = DEMDataset(r'/home/fgao/LoadBalance/PeakPoint/model/test10/train.csv',
                      r'/home/fgao/LoadBalance/PeakPoint/model/test10/sample_image',
                      transform=transforms.Compose([
                          #Rescale(256),
                          #RandomCrop(224),
                          ToTensor(),
                          Normalize(mean, std)
                      ]))
testset = DEMDataset(r'/home/fgao/LoadBalance/PeakPoint/model/test10/test.csv',
                     r'/home/fgao/LoadBalance/PeakPoint/model/test10/sample_image',
                     transform=transforms.Compose([
                         #Rescale(256),
                         #RandomCrop(224),
                         ToTensor(),
                         Normalize(mean, std)
                      ]))
image_datasets = {'train': trainset, 'val': testset}

trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)
dataloader = {'train': trainloader, 'val': testloader}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

# 在已有模型基础上, 重新训练整个网络, 修改全连接层输出, require_grad保持true不变
model_ft = models.resnet18(pretrained=False)
# only train full connected layer parameters
# for params in model_ft.parameters():
#     params.requires_grad = False

print(model_ft)

model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #调整输入通道为1
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 1) #调整输出为1
model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.Linear(256, 128),
    nn.Linear(128, 1)
)

print(model_ft)

criterion = nn.MSELoss()

# retrain all parameters
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# only train full connected layer parameters
# optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.0001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) #优化超参数学习率

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epoches=30)

# save model
torch.save(model_ft.state_dict(), "./result_model/resnet4.pth")
print(model_ft.state_dict())

