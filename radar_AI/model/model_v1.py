import numpy as np
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=0, extractor=0):
        super(EfficientNetClassifier, self).__init__()
        
        self.model = getattr(models, model_name)(pretrained=True)
        self.num_classes = num_classes
        self.extractor = extractor
        
        
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        
        if num_classes == 0:
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1) 
        else:
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes) 

    def forward(self, x):
        if self.extractor == 1:
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = x.flatten(start_dim=1)
            return x
        else:
            x = self.model(x)
            return x

def model_v2(device, model_name='efficientnet_b0', num_classes=0):
    model = EfficientNetClassifier(model_name=model_name, num_classes=num_classes)
    model.to(device)
    return model



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 0, extractor = 0):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Conv layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if (num_classes == 0):
            self.fc = nn.Linear(128 * block.expansion, 1)  # Regression output
        else:
            self.fc = nn.Linear(128 * block.expansion, num_classes)  # num_classes로 수정

        self.extractor = extractor

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if (self.extractor == 1):
            return x
        x = self.fc(x)

        return x

class SleepStageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SleepStageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # h_0과 c_0은 LSTM의 초기 hidden state와 cell state
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # LSTM의 출력
        out, _ = self.lstm(x, (h_0, c_0))
        
        # 마지막 타임스텝의 출력을 FC 레이어에 통과시킴
        out_last = out[:, -1, :]
        out = self.fc(out_last)
        
        return out

class SleepStageGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SleepStageGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU로 변경
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # h_0은 GRU의 초기 hidden state
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        
        # GRU의 출력
        out, _ = self.gru(x, h_0)
        
        # 마지막 타임스텝의 출력을 FC 레이어에 통과시킴
        out_last = out[:, -1, :]
        out = self.fc(out_last)
        
        return out



# ResNet 모델 생성
def model_v1(device, extractor = 0, num_classes = 0):
    model = ResNet(BasicBlock, [2, 2], num_classes, extractor)
    model.to(device)
    return model

# Sleep Stage Classifier 모델 생성
def create_sleep_stage_classifier(device, input_size = 256, hidden_size=64, num_layers=2, num_classes=5):
    model = SleepStageLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    model.to(device)
    return model