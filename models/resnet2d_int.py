import torch
import torch.nn as nn
import torchvision.models as models
import pdb
import models.resnet as resnet

# modules[0]: conv2d (stride=2, res=128)
# modules[1]: batchnorm
# modules[2]: relu
# modules[3]: maxpool (stride=2, res=64)
# modules[4]: 1st conv block - output: C2 (256 channels)
# modules[5]: 2nd conv block (stride=2, res=32) - output: C3 (512 channels)
# modules[6]: 3rd conv block (stride=2, res=16) - output: C4 (1024 channels)
# modules[7]: 4th conv block (stride=2, res=8) - output: C5 (2048 channels)
# modules[8]: 1st deconv block (res=16)
# modules[9]: 2nd deconv block (res=32)
# modules[10]: 3rd deconv block (res=64)
# modules[11]: regression layer

class ResNet152Int(nn.Module):
    def __init__(self, num_joints):
        super(ResNet152Int, self).__init__()
        
        self.num_joints = num_joints

        # Load pretrained resnet-152 model
        pretrained = models.resnet152(pretrained=True)
        
        # Remove last 2 layers
        modules = list(pretrained.children())[:-2]
        
        # Add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer
        modules.append(nn.Conv2d(256, num_joints, 1))

        self.module = nn.ModuleList(modules)

        # For integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(48.0)*4.0+2.0)
        self.wx = self.wx.reshape(1,48).repeat(64,1).reshape(64*48,1)
        self.register_buffer('wy', torch.arange(64.0)*4.0+2.0)
        self.wy = self.wy.reshape(64,1).repeat(1,48).reshape(64*48,1)

    def forward(self, x):
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)
        H = self.module[8](C5)
        H = self.module[9](H)
        H = self.module[10](H)
        H = self.module[11](H)

        #hmap = torch.tensor(H)
        hmap = H.clone()
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64*48))
        hmap = self.relu(hmap) # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6) # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x,y), 2)

        return [H, coord]

class ResNet101Int(nn.Module):
    def __init__(self, num_joints):
        super(ResNet101Int, self).__init__()
        
        self.num_joints = num_joints

        # Load pretrained resnet-101 model
        pretrained = models.resnet101(pretrained=True)
        
        # Remove last 2 layers
        modules = list(pretrained.children())[:-2]
        
        # Add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer
        modules.append(nn.Conv2d(256, num_joints, 1))

        self.module = nn.ModuleList(modules)

        # For integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(48.0)*4.0+2.0)
        self.wx = self.wx.reshape(1,48).repeat(64,1).reshape(64*48,1)
        self.register_buffer('wy', torch.arange(64.0)*4.0+2.0)
        self.wy = self.wy.reshape(64,1).repeat(1,48).reshape(64*48,1)

    def forward(self, x):
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)
        H = self.module[8](C5)
        H = self.module[9](H)
        H = self.module[10](H)
        H = self.module[11](H)

        hmap = torch.tensor(H)
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_joints, 64*48))
        hmap = self.relu(hmap) # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6) # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x,y), 2)

        return [H, coord]


class ResNet50Int(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet50Int, self).__init__()

        self.num_landmarks = num_landmarks

        # Load pretrained resnet-50 model
        pretrained = models.resnet50(pretrained=True)

        # Remove last 2 layers
        modules = list(pretrained.children())[:-2]

        # load resnet with gcnet
        # self.encoder = resnet.resnet50(pretrained=True)
        # modules = nn.ModuleList([])

        # Add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

        # For integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0) * 4.0 + 2.0)
        self.wx = self.wx.reshape(1, 64).repeat(64, 1).reshape(64 * 64, 1)
        self.register_buffer('wy', torch.arange(64.0) * 4.0 + 2.0)
        self.wy = self.wy.reshape(64, 1).repeat(1, 64).reshape(64 * 64, 1)

        self.fliptest = False

    def forward(self, x):
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)
        H = self.module[8](C5)
        H = self.module[9](H)
        H = self.module[10](H)
        H = self.module[11](H)

        hmap = torch.tensor(H)
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_landmarks, 64 * 64))
        hmap = self.relu(hmap)  # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6)  # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x, y), 2)

        return [H, coord]

class ResNet50Int_global(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet50Int_global, self).__init__()
        
        self.num_landmarks = num_landmarks

        # Load pretrained resnet-50 model
        # pretrained = models.resnet50(pretrained=True)
        #
        # # Remove last 2 layers
        # modules = list(pretrained.children())[:-2]

        # load resnet with gcnet
        self.encoder = resnet.resnet50(pretrained=True)
        modules = nn.ModuleList([])

        # Add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

        # For integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0)*4.0+2.0)
        self.wx = self.wx.reshape(1,64).repeat(64,1).reshape(64*64,1)
        self.register_buffer('wy', torch.arange(64.0)*4.0+2.0)
        self.wy = self.wy.reshape(64,1).repeat(1,64).reshape(64*64,1)

        self.fliptest = False

    def forward(self, x):
        B1 = self.encoder(x)
        B2 = self.module[0](B1)
        B3 = self.module[1](B2)
        B4 = self.module[2](B3)
        H = self.module[3](B4)

        hmap = torch.tensor(H)
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_landmarks, 64*64))
        hmap = self.relu(hmap) # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6) # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x,y), 2)

        return [H, coord]

class ResNet34Int(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet34Int, self).__init__()
        
        self.num_landmarks = num_landmarks

        # Load pretrained resnet-34 model
        pretrained = models.resnet34(pretrained=True)
        
        # Remove last 2 layers
        modules = list(pretrained.children())[:-2]
        
        # Add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

        # For integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0)*4.0+2.0)
        self.wx = self.wx.reshape(1,64).repeat(64,1).reshape(64*64,1)
        self.register_buffer('wy', torch.arange(64.0)*4.0+2.0)
        self.wy = self.wy.reshape(64,1).repeat(1,64).reshape(64*64,1)

        self.fliptest = False

    def forward(self, x):
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)
        H = self.module[8](C5)
        H = self.module[9](H)
        H = self.module[10](H)
        H = self.module[11](H)

        hmap = torch.tensor(H)
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_landmarks, 64*64))
        hmap = self.relu(hmap) # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6) # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x,y), 2)

        return [H, coord]

class ResNet18Int(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet18Int, self).__init__()
        
        self.num_landmarks = num_landmarks

        # Load pretrained resnet-18 model
        pretrained = models.resnet18(pretrained=True)
        
        # Remove last 2 layers
        modules = list(pretrained.children())[:-2]
        
        # Add 1st deconv block (stride = 16)
        modules.append(nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 2nd deconv block (stride = 32)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add 3rd deconv block (stride = 64)
        modules.append(nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))

        # Add regression layer
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

        # For integration
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('wx', torch.arange(64.0)*4.0+2.0)
        self.wx = self.wx.reshape(1,64).repeat(64,1).reshape(64*64,1)
        self.register_buffer('wy', torch.arange(64.0)*4.0+2.0)
        self.wy = self.wy.reshape(64,1).repeat(1,64).reshape(64*64,1)

        self.fliptest = False

    def forward(self, x):
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)
        H = self.module[8](C5)
        H = self.module[9](H)
        H = self.module[10](H)
        H = self.module[11](H)

        hmap = torch.tensor(H)
        num_batch = hmap.shape[0]
        hmap = torch.reshape(hmap, (num_batch, self.num_landmarks, 64*64))
        hmap = self.relu(hmap) # For numerical stability
        denom = torch.sum(hmap, 2, keepdim=True).add_(1e-6) # For numerical stability
        hmap = hmap / denom

        x = torch.matmul(hmap, self.wx)
        y = torch.matmul(hmap, self.wy)
        coord = torch.cat((x,y), 2)

        return [H, coord]

