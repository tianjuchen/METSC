import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入模型和损失函数
from sem_vit_hall_petch import SEMViT, HallPetchLoss

class SEMGrainDataset(Dataset):
    """SEM晶粒图像数据集，包含图像、晶粒尺寸分布和屈服强度"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_files[idx])
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        
        # 假设metadata文件与图像文件同名，但扩展名为.npz
        metadata_path = os.path.splitext(img_path)[0] + '.npz'
        metadata = np.load(metadata_path)
        
        # 获取晶粒尺寸分布和屈服强度
        grain_size_dist = torch.tensor(metadata['grain_size_dist'], dtype=torch.float32)
        yield_strength = torch.tensor(metadata['yield_strength'], dtype=torch.float32).view(1)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'grain_size_distribution': grain_size_dist,
            'yield_strength': yield_strength
        }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """训练SEMVIT模型"""
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = batch['image'].to(device)
            targets = {
                'grain_size_distribution': batch['grain_size_distribution'].to(device),
                'yield_strength': batch['yield_strength'].to(device)
            }
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = batch['image'].to(device)
                targets = {
                    'grain_size_distribution': batch['grain_size_distribution'].to(device),
                    'yield_strength': batch['yield_strength'].to(device)
                }
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        # 学习率调整
        scheduler.step(epoch_val_loss)
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_sem_vit_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return history

def visualize_training_history(history):
    """可视化训练历史"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    grain_size_errors = []
    strength_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = {
                'grain_size_distribution': batch['grain_size_distribution'].to(device),
                'yield_strength': batch['yield_strength'].to(device)
            }
            
            outputs = model(images)
            
            # 计算平均晶粒尺寸误差
            pred_size = outputs['mean_grain_size'].cpu().numpy()
            target_size = torch.sum(targets['grain_size_distribution'] * 
                                   torch.linspace(1e-6, 100e-6, 64).to(device), 
                                   dim=1, keepdim=True).cpu().numpy()
            grain_size_errors.append(np.abs(pred_size - target_size))
            
            # 计算屈服强度误差
            pred_strength = outputs['yield_strength'].cpu().numpy()
            target_strength = targets['yield_strength'].cpu().numpy()
            strength_errors.append(np.abs(pred_strength - target_strength))
    
    # 计算平均误差
    mean_size_error = np.mean(np.concatenate(grain_size_errors)) * 1e6  # 转换为微米
    mean_strength_error = np.mean(np.concatenate(strength_errors))
    
    print(f'Mean Grain Size Error: {mean_size_error:.2f} μm')
    print(f'Mean Yield Strength Error: {mean_strength_error:.2f} MPa')
    
    return {
        'grain_size_error': mean_size_error,
        'strength_error': mean_strength_error
    }

def main():
    # 设置参数
    config = {
        'image_size': 224,
        'patch_size': 16,
        'dim': 768,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 3072,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'alpha': 0.5,  # 损失函数中晶粒尺寸和强度的权重
        'data_dir': 'path/to/your/data',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 假设SEM图像是灰度图
    ])
    
    # 创建数据集和数据加载器
    dataset = SEMGrainDataset(root_dir=config['data_dir'], transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 初始化模型、损失函数和优化器
    model = SEMViT(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout'],
        emb_dropout=config['emb_dropout']
    ).to(config['device'])
    
    criterion = HallPetchLoss(alpha=config['alpha'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    print("开始训练模型...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=config['device']
    )
    
    # 可视化训练历史
    visualize_training_history(history)
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_sem_vit_model.pth'))
    print("\n在测试集上评估模型性能:")
    metrics = evaluate_model(model, test_loader, config['device'])
    
    # 打印最终的Hall-Petch参数
    sigma_0 = model.sigma_0.item()
    k = model.k.item()
    print(f"\n学习到的Hall-Petch参数:")
    print(f"摩擦应力 σ₀ = {sigma_0:.4f} MPa")
    print(f"强化系数 k = {k:.4f} MPa·μm^0.5")
    
    # 可视化预测结果与真实值的对比
    visualize_predictions(model, test_loader, config['device'])

def visualize_predictions(model, test_loader, device):
    """可视化模型预测结果与真实值的对比"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['yield_strength'].cpu().numpy()
            
            outputs = model(images)
            predictions = outputs['yield_strength'].cpu().numpy()
            
            # 收集样本用于可视化
            for i in range(len(images)):
                samples.append({
                    'image': images[i].cpu(),
                    'predicted': predictions[i][0],
                    'actual': targets[i][0]
                })
            
            # 只取前几个样本
            if len(samples) >= 9:
                break
    
    # 可视化
    plt.figure(figsize=(12, 12))
    for i, sample in enumerate(samples[:9]):
        plt.subplot(3, 3, i+1)
        plt.imshow(sample['image'][0], cmap='gray')
        plt.title(f"预测: {sample['predicted']:.2f} MPa\n实际: {sample['actual']:.2f} MPa")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.show()

if __name__ == "__main__":
    main()