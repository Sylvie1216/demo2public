#import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

#define
class OASISDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        if not self.image_files:
            raise ValueError(f"No image files found in {data_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

# define VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # 输出均值和对数方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()  # 输出图像像素值 [0,1] 之间
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# 3. 定义损失函数
def loss_function(recon_x, x, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss

# 4. 数据预处理和加载
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
data_dir = 'D:/HuaweiMoveData/Users/HUAWEI/Desktop/keras_png_slices_data/keras_png_slices_seg_train'
dataset = OASISDataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 5. 初始化模型、优化器
vae = VAE(input_dim=64*64, latent_dim=2)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 6. 训练模型
vae.train()
for epoch in range(10):
    total_loss = 0
    for images in dataloader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        recon_images, mu, log_var = vae(images)
        loss = loss_function(recon_images, images, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset):.4f}")

# 7. 可视化潜在空间
vae.eval()
latent_vectors = np.concatenate([vae(images.view(images.size(0), -1))[1].cpu().detach().numpy() for images in dataloader])
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], s=2)
plt.title('Latent Space Visualization')
plt.xlabel('Z1')
plt.ylabel('Z2')
plt.show()

# 8. 生成新图像
with torch.no_grad():
    z = torch.randn(16, 2)
    generated_images = vae.decoder(z).view(-1, 64, 64)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()
