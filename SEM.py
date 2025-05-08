import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 数据加载和预处理
class SEMImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append((img_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')  # 转换为灰度图像
        if self.transform:
            image = self.transform(image)
        return image, label

# Vision Transformer 块
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_threshold=0.1):
        super(SparseMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.sparsity_threshold = sparsity_threshold

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        # 稀疏化注意力图
        attn = torch.where(attn > self.sparsity_threshold, attn, torch.zeros_like(attn))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, sparsity_threshold=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SparseMultiHeadAttention(embed_dim, num_heads, sparsity_threshold)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 基于 ViT 的镍基高温合金 SEM 图像识别模型
class SEMViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=1, num_classes=3,
                 embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, dropout=0.1, sparsity_threshold=0.1):
        super(SEMViT, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, sparsity_threshold)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_output = x[:, 0]
        output = self.head(cls_token_output)
        return output

# 训练和评估代码示例
def train_model(model, train_loader, criterion, optimizer, device, l1_lambda=0.001):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 添加 L1 正则化
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss + l1_lambda * l1_reg
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, recall, f1

# 主函数示例
def main():
    # 数据路径
    data_root = "/mnt/c/Users/Admin/Desktop/METSC"
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()
    ])

    # 划分训练集和测试集
    train_dataset = SEMImageDataset(os.path.join(data_root, 'train'), transform=transform)
    test_dataset = SEMImageDataset(os.path.join(data_root, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEMViT().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 200
    l1_lambda = 0.001
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, l1_lambda)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

    # 评估模型
    accuracy, recall, f1 = evaluate_model(model, test_loader, device)
    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

if __name__ == "__main__":

    create_path = False
    if create_path:
        # 主文件夹路径
        main_folder = "/mnt/c/Users/Admin/Desktop/METSC"

        # 训练集和测试集文件夹
        train_folder = os.path.join(main_folder, "train")
        test_folder = os.path.join(main_folder, "test")

        # 类别列表
        classes = ["class_1", "class_2", "class_3"]

        # 创建主文件夹
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)

        # 创建训练集文件夹和类别子文件夹
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        for class_name in classes:
            class_folder = os.path.join(train_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

        # 创建测试集文件夹和类别子文件夹
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        for class_name in classes:
            class_folder = os.path.join(test_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

        print("目录结构创建完成！")
        print("")
        sys.exit("stop")
    else:
        main()
    