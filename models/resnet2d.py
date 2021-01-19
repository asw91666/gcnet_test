import torchvision.models as models
import torch.nn as nn
import pdb

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

class ResNet152(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet152, self).__init__()
        
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
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

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
        return [H]

class ResNet101(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet101, self).__init__()
        
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
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

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
        return [H]

class ResNet50(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet50, self).__init__()
        
        # Load pretrained resnet-50 model
        pretrained = models.resnet50(pretrained=True)
        
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
        modules.append(nn.Conv2d(256, num_landmarks, 1))

        self.module = nn.ModuleList(modules)

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
        return [H]

class ResNet34(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet34, self).__init__()
        
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
        return [H]

class ResNet18(nn.Module):
    def __init__(self, num_landmarks):
        super(ResNet18, self).__init__()
        
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
        return [H]

