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

class Configtrain:
    def __init__(self):
        self.data_dir = './datafetal/labeled_traindata'            # 数据集路径/labeled_data
        self.batch_size = 16  
        self.num_epochs = 50  
        self.learning_rate = 1e-4  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 3  
        self.output_channels = 2  
        self.img_size = (256, 256)  
        self.checkpoint_dir = './checkpoints'  
        self.log_dir = './logs'  

class Configval:
    def __init__(self):
        self.data_dir = './datafetal/labeled_valdata'            
        self.batch_size = 16  
        self.num_epochs = 50  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 3  
        self.output_channels = 2  
        self.img_size = (256, 256)    

configtrain = Configtrain()
configval = Configval()

transformtrain = transforms.Compose([
    transforms.Resize(configtrain.img_size),
    transforms.ToTensor(),
])

transformval = transforms.Compose([
    transforms.Resize(configval.img_size),
    transforms.ToTensor(),
])
train_dataset = FetalDataset(configtrain.data_dir, transform=transformtrain)
val_dataset = FetalDataset(configval.data_dir, transform=transformval) 
train_loader = DataLoader(train_dataset, batch_size=configtrain.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configval.batch_size, shuffle=True )

# 创建模型
model = nn.Sequential(
    Encoder(configtrain),
    Decoder(configtrain)
)
model.to(configtrain.device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于分类任务的交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=configtrain.learning_rate)

# TensorBoard 日志
writer = SummaryWriter(configtrain.log_dir)

# 训练循环
best_val_loss = float('inf')  # 用于保存最佳验证损失
best_model_wts = None         # 保存最优模型的权重
for epoch in range(configtrain.num_epochs):
    model.train()  
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(configtrain.device), labels.to(configtrain.device)

        # 前向传播
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{configtrain.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 计算平均损失
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{configtrain.num_epochs}], Average Loss: {avg_loss:.4f}')

    # 写入 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(configtrain.device), val_labels.to(configtrain.device)

                # 前向传播
                val_outputs = model(val_images)
                val_loss_batch = criterion(val_outputs, val_labels)

                val_loss += val_loss_batch.item()
        checkpoint_path = os.path.join(configtrain.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

# 关闭 TensorBoard
writer.close()