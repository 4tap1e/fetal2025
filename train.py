import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pre import FetalDataset  
from unet import Encoder, Decoder  
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

class Config:
    def __init__(self):
        self.data_dir = './datafetal'  # 数据集路径
        self.batch_size = 16  
        self.num_epochs = 50  
        self.learning_rate = 1e-4  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 3  
        self.output_channels = 2  
        self.img_size = (256, 256)  
        self.checkpoint_dir = './checkpoints'  
        self.log_dir = './logs'  

config = Config()

transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
])

train_dataset = FetalDataset(config.data_dir, transform=transform)  # 假设你有一个自定义数据集
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# 创建模型
model = nn.Sequential(
    Encoder(config),
    Decoder(config)
)
model.to(config.device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于分类任务的交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# TensorBoard 日志
writer = SummaryWriter(config.log_dir)

# 训练循环
for epoch in range(config.num_epochs):
    model.train()  
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(config.device), labels.to(config.device)

        # 前向传播
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 计算平均损失
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {avg_loss:.4f}')

    # 写入 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

# 关闭 TensorBoard
writer.close()