import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一调整为224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 自定义数据集类（与您的ViT代码保持一致）
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

# 加载并修改VGG模型
def load_vgg_model(model_path):
    # 加载预训练的VGG16模型
    model = models.vgg16(pretrained=False)
    
    # 修改分类器部分以匹配保存的模型结构
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    
    # 加载预训练权重（允许部分匹配）
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    
    # 移除最后的分类层，保留特征提取部分
    model.classifier[6] = nn.Identity()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    model.eval()
    return model

# 加载并修改ViT模型（与您的训练代码完全兼容）
def load_vit_model(model_path):
    # 创建与训练代码完全相同的模型结构
    class ViTForFeatureExtraction(nn.Module):
        def __init__(self):
            super(ViTForFeatureExtraction, self).__init__()
            # 使用与训练代码相同的ViT初始化方式
            from modelscope import snapshot_download
            model_dir = snapshot_download("google/vit-base-patch16-224-in21k")
            self.vit = ViTModel.from_pretrained(model_dir)
            # 保留分类头结构但不使用（为了正确加载权重）
            self.classifier = nn.Sequential(
                nn.Linear(self.vit.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
            
        def forward(self, x):
            outputs = self.vit(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]  # 只返回[CLS] token特征
    
    # 初始化模型
    model = ViTForFeatureExtraction().to(device)
    
    # 加载预训练权重
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)  # 这次需要严格匹配
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    return model

# 修改LLMClassifier类定义
class LLMClassifier(nn.Module):
    def __init__(self, llm_path, feature_dim, num_classes=2):
        super(LLMClassifier, self).__init__()
        
        # 加载预训练的LLM
        self.llm = AutoModel.from_pretrained(llm_path)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # 应用LoRA
        self.llm = get_peft_model(self.llm, lora_config)
        
        # 特征投影层
        self.feature_proj = nn.Linear(feature_dim, self.llm.config.hidden_size)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, features):
        # 投影特征到LLM的隐藏空间
        projected_features = self.feature_proj(features)
        
        # 通过LLM处理特征
        outputs = self.llm(inputs_embeds=projected_features.unsqueeze(1))
        
        # 使用[CLS] token或第一个位置的输出
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 分类
        logits = self.classifier(pooled_output)
        return logits
    
    # 添加打印可训练参数的方法
    def print_trainable_parameters(self):
        self.llm.print_trainable_parameters()

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, vgg_model, vit_model):
        super(FeatureExtractor, self).__init__()
        self.vgg = vgg_model
        self.vit = vit_model
        
    def forward(self, x):
        # VGG特征提取
        vgg_features = self.vgg(x)
        
        # ViT特征提取
        vit_features = self.vit(x)
        
        # 拼接特征
        combined_features = torch.cat([vgg_features, vit_features], dim=1)
        return combined_features

# 训练函数
# def train_model(feature_extractor, classifier, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
#     best_val_acc = 0.0
#     train_loss_history = []
#     val_loss_history = []
#     train_acc_history = []
#     val_acc_history = []
    
#     for epoch in range(num_epochs):
#         feature_extractor.eval()  # 特征提取器始终在eval模式
#         classifier.train()
        
#         running_loss = 0.0
#         running_corrects = 0
        
#         total_batches = len(train_loader)
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print("-" * 30)
        
#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()
            
#             # 提取特征
#             with torch.no_grad():
#                 features = feature_extractor(inputs)
            
#             # 分类
#             outputs = classifier(features)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
            
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
            
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
            
#             if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
#                 print(f"Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")
        
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)
#         train_loss_history.append(epoch_loss)
#         train_acc_history.append(epoch_acc.cpu().numpy())
        
#         # 验证阶段
#         val_loss, val_acc = evaluate_model(feature_extractor, classifier, val_loader, criterion)
#         val_loss_history.append(val_loss)
#         val_acc_history.append(val_acc)
        
#         print(f"\nEpoch {epoch+1} Summary:")
#         print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
#         print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
#         print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
#         # 保存最佳模型
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(classifier.state_dict(), '/root/LLM/Model/best_llm_classifier.pth')
#             print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
    
#     # 绘制训练曲线
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss_history, label='Train Loss')
#     plt.plot(val_loss_history, label='Val Loss')
#     plt.legend()
#     plt.title('Loss History')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc_history, label='Train Acc')
#     plt.plot(val_acc_history, label='Val Acc')
#     plt.legend()
#     plt.title('Accuracy History')
    
#     plt.savefig('llm_training_history.png')
#     plt.close()
    
#     return classifier

# 评估函数
def evaluate_model(feature_extractor, classifier, data_loader, criterion):
    feature_extractor.eval()
    classifier.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 提取特征
            features = feature_extractor(inputs)
            
            # 分类
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    total_loss = running_loss / len(data_loader.dataset)
    total_acc = running_corrects.double() / len(data_loader.dataset)
    
    return total_loss, total_acc.cpu().numpy()

def train_model(feature_extractor, classifier, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=3):
    best_val_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    test_acc_history = []  # 记录测试集准确率
    
    for epoch in range(num_epochs):
        feature_extractor.eval()  # 特征提取器始终在eval模式
        classifier.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        total_batches = len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 提取特征
            with torch.no_grad():
                features = feature_extractor(inputs)
            
            # 分类
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.cpu().numpy())
        
        # 验证阶段
        val_loss, val_acc = evaluate_model(feature_extractor, classifier, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # if epoch == 0:
        #     continue
        # 保存最佳模型并在测试集上评估
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), '/root/LLM/Model/best_llm_classifier3.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")
            
            # 在测试集上评估当前最佳模型
            test_loss, test_acc = evaluate_model(feature_extractor, classifier, test_loader, criterion)
            test_acc_history.append(test_acc)
            print(f"Test Acc with current best model: {test_acc:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    if test_acc_history:  # 如果有测试集结果，也绘制出来
        plt.plot([i for i, val in enumerate(val_acc_history) if val == max(val_acc_history[:i+1])][1:], 
                 test_acc_history, 'o-', label='Test Acc (on best val)')
    plt.legend()
    plt.title('Accuracy History')
    
    plt.savefig('llm_training_history.png')
    plt.close()
    
    return classifier

# 主函数
def main():
    # 数据集路径
    train_dir = '/root/LLM/data/training_set_limited'
    val_dir = '/root/LLM/data/validation_set_limited'
    test_dir = '/root/LLM/data/test_set_limited'
    
    # 创建数据集
    train_dataset = DeepfakeDataset(train_dir, transform=transform, max_samples_per_class=100)
    val_dataset = DeepfakeDataset(val_dir, transform=transform, max_samples_per_class=100)
    test_dataset = DeepfakeDataset(test_dir, transform=transform, max_samples_per_class=100)
    
    # 创建数据加载器
    batch_size = 16  # 减小batch size以适应LLM
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # 加载预训练模型
    print("Loading VGG model...")
    vgg_model = load_vgg_model('/root/LLM/Model/best_VGG_model.pth')
    
    print("Loading ViT model...")
    vit_model = load_vit_model('/root/LLM/Model/best_vit_model.pth')
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(vgg_model, vit_model).to(device)
    
    # 计算特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        vgg_feature = vgg_model(dummy_input)
        vit_feature = vit_model(dummy_input)
        feature_dim = vgg_feature.shape[1] + vit_feature.shape[1]
        print(f"Combined feature dimension: {feature_dim}")
    
    # 初始化LLM分类器
    print("Initializing LLM classifier...")
    llm_path = '/root/LLM/Model/Qwen3-Embedding-8B'
    classifier = LLMClassifier(llm_path, feature_dim).to(device)
    
    # 打印可训练参数
    classifier.print_trainable_parameters()
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 学习率调度器
    total_steps = len(train_loader) * 15
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练模型
    # 训练模型
    print("Start training...")
    classifier = train_model(
        feature_extractor, classifier, train_loader, val_loader, test_loader,  # 添加test_loader
        criterion, optimizer, scheduler, num_epochs=2
    )
    
    # 加载最佳模型
    classifier.load_state_dict(torch.load('/root/LLM/Model/best_llm_classifier3.pth'))
    
    # 在测试集上最终评估
    test_loss, test_acc = evaluate_model(feature_extractor, classifier, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # 保存模型
    torch.save(classifier.state_dict(), '/root/LLM/Model/final_llm_classifier3.pth')
    torch.save(classifier, '/root/LLM/Model/full_llm_classifier3.pth')

if __name__ == '__main__':
    main()