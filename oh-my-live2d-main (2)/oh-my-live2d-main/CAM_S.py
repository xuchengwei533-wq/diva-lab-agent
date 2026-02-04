import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.autograd
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import logging
import shutil
from torchlibrosa.augmentation import SpecAugmentation

random.seed(1314)
np.random.seed(1314)
torch.manual_seed(1314)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1314)

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)

        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)

        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)

        aggregate_weight = torch.sum(aggregate_weight, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (state['log_dir'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (state['log_dir']) + 'model_best.pth')

class CustomDataset(Dataset):
    def __init__(self, data_dir, train=False, val=False, transforms=None):
        self.data_dir = data_dir
        self.MFCC_data_dir = os.path.join(self.data_dir, 'MFCC_Output')
        self.Label_data_dir = os.path.join(self.data_dir, 'Label')

        # 获取所有文件列表
        mfcc_files = os.listdir(self.MFCC_data_dir)
        label_files = os.listdir(self.Label_data_dir)

        # 构建以样本ID为键的字典
        mfcc_dict = {
            f.replace('_MFCC.xlsx', ''): f
            for f in mfcc_files if f.endswith('_MFCC.xlsx')
        }

        label_dict = {
            f.replace('.xlsx', ''): f
            for f in label_files if f.endswith('.xlsx')
        }

        # 找到两边都有的样本
        common_keys = list(set(mfcc_dict.keys()) & set(label_dict.keys()))
        common_keys.sort()  # 可选，保证一致性

        # 构建匹配对列表
        self.file_pairs = [(mfcc_dict[k], label_dict[k]) for k in common_keys]

        # 打乱顺序（可以设置随机种子确保一致性）
        random.seed(1314)
        random.shuffle(self.file_pairs)

        # 数据集划分
        total_len = len(self.file_pairs)
        if train:
            self.file_pairs = self.file_pairs[:int(0.8 * total_len)]
        elif val:
            self.file_pairs = self.file_pairs[int(0.8 * total_len):]

        self.transforms = transforms

        print("【调试】前5个样本配对如下：")
        for i in range(min(5, len(self.file_pairs))):
            print(f"  MFCC 文件: {self.file_pairs[i][0]}  <==>  Label 文件: {self.file_pairs[i][1]}")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        MFCC_file_name, Label_file_name = self.file_pairs[idx]

        MFCC_file_path = os.path.join(self.MFCC_data_dir, MFCC_file_name)
        Label_file_path = os.path.join(self.Label_data_dir, Label_file_name)

        # 读取 MFCC 数据
        MFCC_data = pd.read_excel(MFCC_file_path, header=None, engine="openpyxl").values.astype(float)
        MFCC_tensor = torch.tensor(MFCC_data, dtype=torch.float32).unsqueeze(0)

        # 读取标签数据
        Label_dataframe = pd.read_excel(Label_file_path)
        label = Label_dataframe.values[:, 1:].astype(float)[:10, :].ravel()
        Label_tensor = torch.tensor(label - 1, dtype=torch.float32)

        if self.transforms is not None:
            MFCC_tensor = self.transforms(MFCC_tensor)

        return MFCC_tensor, Label_tensor



def train_epoch(net, train_loader, criterion, optimizer, device, lr_scheduler):
    net.train()
    train_loss = []
    correct = 0
    total = 0
    for im, label in train_loader:
        im, label = im.to(device), label.to(device)
        optimizer.zero_grad()

        output, _, _ = net(im)
        output = output.view(output.shape[0], 5, 10)

        loss = criterion(output.float(), label.long())
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        total += label.size(0) * label.size(1)
        correct += (label == output.argmax(dim=1)).sum().item()

    return np.mean(train_loss), correct / total


def validate(net, val_loader, criterion, device):
    net.eval()
    val_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for im, label in val_loader:
            im, label = im.to(device), label.to(device)
            output, _, _ = net(im)
            output = output.view(output.shape[0], 5, 10)
            loss = criterion(output.float(), label.long())
            val_loss.append(loss.item())
            total += label.size(0) * label.size(1)
            correct += (label == output.argmax(dim=1)).sum().item()

    return np.mean(val_loss), correct / total


def train_model(net, train_loader, val_loader, num_epochs, optimizer, criterion, lr_scheduler, device, log_dir, pretrained_weights=None):
    if pretrained_weights and os.path.exists(pretrained_weights):
        net.load_state_dict(torch.load(pretrained_weights))
        logging.info(f"Loaded pretrained weights from {pretrained_weights}")

    writer = SummaryWriter(log_dir=log_dir)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, device, lr_scheduler)
        val_loss, val_acc = validate(net, val_loader, criterion, device)
        lr_scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Val_Loss', val_loss, epoch)
        writer.add_scalar('Val_Accuracy', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            model_path = os.path.join(log_dir, 'best_model.pth')

            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(net.state_dict(), model_path)
                logging.warning(f"Saved best model with validation accuracy: {val_acc:.4f}")
            except Exception as e:
                logging.error(f"Failed to save best model: {e}")
                break

    writer.close()

def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU())
        elif name == 'sigmod':
            nonlinear.add_module('sigmod', nn.Sigmoid())
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm2d(channels))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear

def statistics_pooling(x, axis=1, keepdim=True, unbiased=True):
    mean = x.mean(dim=axis)
    std = x.std(dim=axis, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=1)
    if keepdim:
        stats = stats.unsqueeze(dim=axis)
    return stats

class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)

class ODConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=2, padding=1, dilation=1, kernel_num=1, reduction=0.0625,
                 config_str='batchnorm-relu'):
        super(ODConvLayer, self).__init__()

        self.linear = ODConv2d(in_channels, out_channels,
                               kernel_size, stride=stride,
                               padding=padding, dilation=dilation,
                               groups=1, kernel_num=kernel_num, reduction=reduction)
        self.nonlinear1= get_nonlinear(config_str, out_channels)


    def forward(self, x):
        x = self.linear(x)
        x1 = self.nonlinear1(x)

        return x1

class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv2d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.linear1 = nn.Conv2d(1, bn_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Conv2d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        x0 = x.max(1, keepdim=True)[0]
        context = x.mean(1, keepdim=True) + x0
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, config_str='batchnorm-relu'):
        super(CAMDenseTDNNLayer, self).__init__()

        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation

        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv2d(in_channels, bn_channels, 1)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x

class CAMDenseTDNNBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, config_str='batchnorm-relu'):
        super(CAMDenseTDNNBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels, out_channels=out_channels,
                                      bn_channels=bn_channels, kernel_size=kernel_size, stride=stride,
                                      dilation=dilation, config_str=config_str)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x

class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str='batchnorm-relu'):
        super(LinearLayer, self).__init__()
        self.linear = nn.Conv2d(in_channels, out_channels, 1)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x

class BasicResBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, expansion=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.expansion = expansion
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=64, in_channels=1):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(in_channels, m_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2, expansion=1)


    def _make_layer(self, block, planes, num_blocks, stride, expansion):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, expansion))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)

        return out

class CAMPPlus(nn.Module):
    def __init__(self, num_class, input_size, embd_dim=4096, growth_rate=32, bn_size=4, in_channels=64, init_channels=128, config_str='batchnorm-relu'):
        super(CAMPPlus, self).__init__()

        self.head = FCM(block=BasicResBlock, num_blocks=[2, 2], m_channels=64, in_channels=input_size)

        self.xvector = nn.Sequential(ODConvLayer(in_channels, in_channels * 2, kernel_size=3, stride=1, dilation=1, padding=1, kernel_num=1, reduction=0.0625,
                                    config_str='batchnorm-relu'),
                                    ODConvLayer(in_channels * 2, in_channels, kernel_size=1, stride=1, dilation=1, padding=0, kernel_num=1, reduction=0.0625,
                                    config_str='batchnorm-relu'))

        self.xvector0 = nn.Sequential(
            ODConvLayer(in_channels, in_channels * 2, kernel_size=3, stride=1, dilation=1, padding=1, kernel_num=1,
                        reduction=0.0625,
                        config_str='batchnorm-relu'),
            ODConvLayer(in_channels * 2, in_channels, kernel_size=1, stride=1, dilation=1, padding=0, kernel_num=1,
                        reduction=0.0625,
                        config_str='batchnorm-relu'))

        self.xvector1 = nn.Sequential(CAMDenseTDNNBlock(num_layers=3, in_channels=in_channels, out_channels=growth_rate, bn_channels=growth_rate * bn_size,
                                                       kernel_size=3, stride=1, dilation=1, config_str=config_str),
                                     TransitLayer(in_channels=init_channels + growth_rate * 2, out_channels=init_channels, bias=False, config_str=config_str))

        self.relu = get_nonlinear(config_str, init_channels)
        self.pool = StatsPool()
        self.xvector_3 = LinearLayer(1, 1, config_str='batchnorm-relu')
        self.output_1 = nn.Linear(embd_dim, num_class)

        self.spec_augmenter = SpecAugmentation(time_drop_width=1, time_stripes_num=16,
            freq_drop_width=1, freq_stripes_num=16)
        self.bn0 = nn.BatchNorm2d(128)

    def forward(self, x): # B, 1, freq, time

        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.head(x)  #B 64 64 64

        x1 = self.xvector(x) #B 64 64 64
        x2 = self.xvector0(x1)

        x3 = self.xvector1(x)
        x3 = self.relu(x3)  #B 128 64 64

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pool(x) #B 1 128 64

        feature = x

        x = self.xvector_3(x)
        x = x.view(x.size(0), -1)

        feature1 = x

        x = self.output_1(x)

        return x, feature, feature1

if __name__ == '__main__':
    data_dir = r"/home/zx/111/sopran"
    train_batch_size = 16
    val_batch_size = 16
    num_workers = 4
    num_classes = 50
    num_epochs = 120
    learning_rate = 4e-5
    pretrained_weights = None

    train_dataset = CustomDataset(data_dir, train=True)
    val_dataset = CustomDataset(data_dir, val=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAMPPlus(num_class=num_classes,
                     input_size=1,
                     embd_dim=8192,
                     growth_rate=64,
                     bn_size=4,
                     init_channels=128,
                     config_str='batchnorm-relu',
                     ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs_ddnet_sopran_2637', current_time)
    os.makedirs(log_dir, exist_ok=True)

    train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, lr_scheduler, device, log_dir,
                pretrained_weights)
    train_dataset = CustomDataset(data_dir, train=True)