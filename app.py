#PASTE THIS CODE IN GOOGLE COLAB NOTEBOOK
#========================================



# ===========================
# Cell 1: Install Dependencies
# ===========================
!pip install torch torchvision nltk pycocotools matplotlib Pillow

# Download NLTK resources
import nltk
nltk.download('punkt')

# ===========================
# Cell 2: Import Libraries
# ===========================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
from nltk.tokenize import word_tokenize
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# Cell 3: Define Generator Model
# ===========================
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 3, 64, 64)

# ===========================
# Cell 4: Define Discriminator Model
# ===========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

# ===========================
# Cell 5: Define Dataset Class
# ===========================
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = word_tokenize(caption.lower())
        encoded = torch.tensor([len(tokens)])
        return image, encoded

    def __len__(self):
        return len(self.ids)

# ===========================
# Cell 6: Set Hyperparameters and Load Data
# ===========================
latent_dim = 100

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = '/content/coco_data/train2017'
annFile = '/content/coco_data/annotations/captions_train2017.json'

dataset = CocoDataset(root, annFile, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# ===========================
# Cell 7: Training Loop
# ===========================
for epoch in range(5):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z).detach()
        d_loss = criterion(discriminator(real_images), real_labels) + \
                 criterion(discriminator(fake_images), fake_labels)
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        generated_images = generator(z)
        g_loss = criterion(discriminator(generated_images), real_labels)
        g_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    save_image(generated_images, f"generated_epoch_{epoch+1}.png")

# ===========================
# Cell 8: Generate an Image from Text
# ===========================
def generate_image():
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        img = generator(z).cpu().squeeze(0)
    return img

img = generate_image()
plt.imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
plt.axis('off')
plt.show()
