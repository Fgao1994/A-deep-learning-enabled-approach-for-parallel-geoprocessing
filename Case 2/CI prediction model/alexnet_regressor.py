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
from torchvision.models import AlexNet
from data_read import DEMDataset
# from dl.peak_point.data_read import DEMDataset
from sklearn.metrics import r2_score
import os
# os.environ['TORCH_HOME']='F:/Python/ml_env/dl/pretrain_model'
os.environ['TORCH_HOME']='/home/fgao/LoadBalance/Intersection/python/ml_env/dl/pretrain_model'
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
        # image = F.normalize(image, self.mean, self.std, self.inplace)
        return {'image': image, 'label': label}

# 函数：trainning
def train_model(model, criterion, optimizer, scheduler, num_epoches=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
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
                labels = data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    labels_array = labels.data.cpu().detach().numpy().flatten()
                    outputs_array = outputs.data.cpu().detach().numpy().flatten()
                    for ele in labels_array: y_true.append(ele)
                    for ele in outputs_array: y_predict.append(ele)
                   
                    # print(f'Loss of batch {i} in epoch {epoch}: {loss.item()}')
                    # if i % 20 == 19:
                    #     print(f'Loss of batch {i} in epoch {epoch}: {loss.item()}')
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        # for name, parms in model.named_parameters():
                        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                        #           torch.mean(parms.data),
                        #           ' -->grad_value:', torch.mean(parms.grad))
                running_loss += loss.item() * inputs.size(0) #这里乘以inputs.size(0),是为了得到批次里样本的loss和，下面的epoch_loss就是每个样本的误差
                # print('running loss:{}'.format(running_loss))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_r2 = r2_score(y_true, y_predict)

#             if phase == "train":
#                 for name, parms in model.named_parameters():
#                     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
#                           ' -->grad_value:', torch.mean(parms.grad))

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
print(torch.version.cuda)
print(torch.cuda.is_available())

# test1
# mean = 3126.413
# std =  401.522

# test2
# mean = 3094.995
# std =  401.218

# peak point case
mean = 2096.526
std = 971.579

#intersection case
mean = 0.15095
std = 0.34627


trainset = DEMDataset(r'/home/fgao/LoadBalance/Intersection/model/test5/train.csv',
                      r'/home/fgao/LoadBalance/Intersection/model/test5/sample_image',
                      transform=transforms.Compose([
                          #Rescale(256),
                          #RandomCrop(224), # 227?
                          ToTensor(),
                          Normalize(mean, std)
                      ]))
testset = DEMDataset(r'/home/fgao/LoadBalance/Intersection/model/test5/test.csv',
                      r'/home/fgao/LoadBalance/Intersection/model/test5/sample_image',
                     transform=transforms.Compose([
                         #Rescale(256),
                         #RandomCrop(224), # 227?
                         ToTensor(),
                         Normalize(mean, std)
                     ]))

# from skimage import io
# for i in range(len(trainset)):
#     sample = trainset[i]
#     img = sample['image']
#     #print(sample['path'])
#     print(img.shape)
#     print(type(img))
#     print(img.min(), img.max(), img.mean())
#     io.imshow(img.view(256, 256, 1).numpy())
#     io.show()

image_datasets = {'train': trainset, 'val': testset}

trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)
dataloader = {'train': trainloader, 'val': testloader}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

# 在已有模型基础上, 重新训练整个网络, 修改全连接层输出, require_grad保持true不变
model_ft: AlexNet = models.alexnet(pretrained=False)
# only train full connected layer parameters
# for params in model_ft.parameters():
#     params.requires_grad = False

print(model_ft)
model_ft.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
model_ft.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 1)
        )
print(model_ft)
criterion = nn.MSELoss()

# retrain all parameters
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)

# only train full connected layer parameters
# optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.0001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) #优化超参数学习率

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epoches=30)

# save model
torch.save(model_ft.state_dict(), "./result_model/alexnet1.pth")
print(model_ft.state_dict())

#######
