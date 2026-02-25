import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 使用PyTorch内置Transformer的实现
class PatchEmbedding(nn.Module):
    """将图像分割为patch并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # (B, C, H, W) -> (B, E, N)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class VisionTransformer(nn.Module):
    """使用PyTorch内置TransformerEncoder的ViT实现"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=192, num_heads=3, num_layers=6, 
                 num_classes=2, dropout=0.1):
        super().__init__()
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # 类别token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 嵌入patches
        x = self.patch_embed(x)  # (B, N, E)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, E)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码器
        # 注意PyTorch Transformer期望的输入形状是(SeqLen, Batch, Embedding)
        x = x.transpose(0, 1)  # (N+1, B, E)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (B, N+1, E)
        
        # 分类
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 只使用类别token
        logits = self.head(cls_token_final)
        
        return logits

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 自定义数据集类
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples_per_class=100):
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples = max_samples_per_class
        self.samples = []
        
        # 加载Deepfakes类样本
        deepfake_dir = os.path.join(root_dir, 'Deepfakes')
        if os.path.exists(deepfake_dir):
            for subdir in os.listdir(deepfake_dir):
                subdir_path = os.path.join(deepfake_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path)[:self.max_samples]:
                        img_path = os.path.join(subdir_path, img_file)
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, 1))  # 1表示Deepfake
        
        # 加载Original类样本
        original_dir = os.path.join(root_dir, 'Original')
        if os.path.exists(original_dir):
            for subdir in os.listdir(original_dir):
                subdir_path = os.path.join(original_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path)[:self.max_samples]:
                        img_path = os.path.join(subdir_path, img_file)
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, 0))  # 0表示Original
        
        # 打乱样本顺序
        np.random.shuffle(self.samples)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    best_val_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 添加进度条
        total_batches = len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 打印每个batch的进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.cpu().numpy())
        
        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '/root/LLM/Model/best_trans_model.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.legend()
    plt.title('Accuracy History')
    
    plt.savefig('vit_training_history.png')
    plt.close()
    
    return model

# 评估函数
def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    total_loss = running_loss / len(data_loader.dataset)
    total_acc = running_corrects.double() / len(data_loader.dataset)
    
    return total_loss, total_acc.cpu().numpy()

# 测试函数
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(cm)
    
    return accuracy, cm

# 主函数
def main():
    # 数据集路径
    train_dir = '/root/LLM/data/training_set_limited'
    val_dir = '/root/LLM/data/validation_set_limited'
    test_dir = '/root/LLM/data/test_set_limited'
    
    # 创建数据集
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform, max_samples_per_class=100)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transform, max_samples_per_class=100)
    test_dataset = DeepfakeDataset(test_dir, transform=val_transform, max_samples_per_class=100)
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # 初始化Vision Transformer模型
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,  # 较小的embedding维度
        num_heads=3,
        num_layers=6,   # 6个Transformer层
        num_classes=2
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    
    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('/root/LLM/Model/best_trans_model.pth'))
    
    # 在测试集上评估
    test_accuracy, cm = test_model(model, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), '/root/LLM/Model/final_trans_model.pth')
    torch.save(model, '/root/LLM/Model/full_trans_model.pth')  # 保存整个模型

if __name__ == '__main__':
    main()