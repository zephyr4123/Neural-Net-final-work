import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from matplotlib import rcParams
from matplotlib import font_manager

# 设置中文字体
font_path = "C:/Windows/Fonts/simhei.ttf"  # 确保路径正确（这里以 Windows 系统为例）
if os.path.exists(font_path):
    rcParams['font.sans-serif'] = ['SimHei']
else:
    print("未找到中文字体，使用默认字体显示")
rcParams['axes.unicode_minus'] = False  # 避免负号显示问题

# 创建保存图片的目录
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

# 超参
latent_dim = 100
image_dim = 28 * 28  # MNIST 图像尺寸
batch_size = 512
lr = 1e-4
n_epochs = 20
critic_iterations = 10  # 每次更新生成器前，更新判别器的次数
lambda_gp = 10  # 梯度惩罚系数

# 生成器
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

# 判别器
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# 梯度惩罚函数
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 数据归一化到 [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
generator = Generator(latent_dim, image_dim)
critic = Critic(image_dim)
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
critic.to(device)

generator_losses = []
critic_losses = []

# 训练
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.view(imgs.size(0), -1).to(device)
        batch_size = real_imgs.size(0)

        # 更新判别器
        for _ in range(critic_iterations):
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z).detach()

            real_validity = critic(real_imgs)
            fake_validity = critic(fake_imgs)

            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
            loss_C = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

        # 更新生成器
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)

        loss_G = -torch.mean(critic(fake_imgs))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    generator_losses.append(loss_G.item())
    critic_losses.append(loss_C.item())

    print(f"轮数 [{epoch+1}/{n_epochs}] 判别器损失: {loss_C.item():.4f} 生成器损失: {loss_G.item():.4f}")

# 生成最终图片并保存
z = torch.randn(64, latent_dim).to(device)
final_imgs = generator(z).view(-1, 1, 28, 28).cpu().detach()
save_image(final_imgs, f"{output_dir}/final_generated_images.png", nrow=8, normalize=True)

# 绘制损失变化曲线并保存
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label="生成器损失")
plt.plot(critic_losses, label="判别器损失")
plt.xlabel("训练轮数")
plt.ylabel("损失")
plt.title("WGAN-GP 损失变化曲线")
plt.legend()
plt.savefig(f"{output_dir}/loss_curve.png")
plt.show()

print(f"训练完成！最终生成的图片保存在 {output_dir}/final_generated_images.png")
