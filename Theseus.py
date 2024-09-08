import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from transformers import GPT2Tokenizer
import os
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F

class RealVideoDataset(Dataset):
    def __init__(self, json_file, video_dir, transform=None, max_length=512):
        with open(json_file, 'r') as f:
          self.data = json.load(f)
        self.video_dir = video_dir
        self.transform = transform
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.data[idx]['video_file'])
        description = self.data[idx]['description']

        # Load and preprocess the video
        frames = self.load_video(video_path)
        if self.transform:
            frames = self.transform(frames)

        # Tokenize the description
        encoded_text = self.tokenizer.encode_plus(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return frames, encoded_text['input_ids'].squeeze(0)

    def load_video(self, video_path, max_frames=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        cap.release()

        # Make sure all videos have the same amount of frames
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        elif len(frames) < max_frames:
            frames.extend([frames[-1]] * (max_frames - len(frames)))  # Padding with the last frame

        return frames


class CausalConv3d(nn.Conv3d):
    """ Causal 3D convolution layer for temporal consistency. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CausalConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.pad = (self.kernel_size[0] - 1, 0, 0, 0, 0, 0)

    def forward(self, x):
        x = nn.functional.pad(x, self.pad)
        return super(CausalConv3d, self).forward(x)


class VAE3D(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super(VAE3D, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_mu = nn.Linear(128 * 4 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 16 * 16, latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 128 * 4 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        print(f"Shape after encoder: {h.shape}")
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(h.size(0), 128, 4, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ExpertTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6):
        super(ExpertTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))  # Average over sequence dimension


class Theseus(nn.Module):
    def __init__(self, vae, transformer):
        super(Theseus, self).__init__()
        self.vae = vae
        self.transformer = transformer
        self.expert_adaln = nn.LayerNorm(1024)  # Adapt for different modalities
        self.text_mapper = nn.Linear(512, 1024)

    def forward(self, x_video, x_text):
        # VAE to compress the video
        mu, logvar = self.vae.encode(x_video)
        z_vision = self.vae.reparameterize(mu, logvar)

        # Transformer to process the text
        z_text = self.transformer(x_text)
        z_text = self.text_mapper(z_text)

        # Normalize and combine vision and text features
        z_vision = self.expert_adaln(z_vision)
        z_combined = z_vision + z_text.unsqueeze(1).unsqueeze(2)  # Adjust for 3D tensors

        # Rebuild the video
        reconstructed_video = self.vae.decode(z_combined)

        return reconstructed_video, mu, logvar

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_layers = nn.ModuleList(vgg.features[:23]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for layer in self.vgg_layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)
        return features

def vae_loss(recon_x, x, mu, logvar, kl_weight=0.0001):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld_loss

def perceptual_loss(recon_x, x, vgg_model):
    b, c, t, h, w = recon_x.shape
    recon_x = recon_x.transpose(1, 2).reshape(b*t, c, h, w)
    x = x.transpose(1, 2).reshape(b*t, c, h, w)
    
    recon_features = vgg_model(recon_x)
    original_features = vgg_model(x)
    
    loss = 0
    for recon_feature, original_feature in zip(recon_features, original_features):
        loss += F.mse_loss(recon_feature, original_feature)
    return loss / t  # Normalizar por el número de frames

def total_loss(recon_x, x, mu, logvar, vgg_model, perceptual_weight=0.1):
    vae_loss_value = vae_loss(recon_x, x, mu, logvar)
    perceptual_loss_value = perceptual_loss(recon_x, x, vgg_model)
    return vae_loss_value + perceptual_weight * perceptual_loss_value


def save_video(tensor, path):
    tensor = tensor.detach().permute(1, 2, 3, 0).cpu().numpy()  # Switch to (T, H, W, C)
    tensor = (tensor * 255).astype(np.uint8)  # Scalar from [0, 1] to [0, 255]

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (tensor.shape[2], tensor.shape[1]))

    for frame in tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def train_vae_only(vae, dataloader, num_epochs, device):
    vae.train()  # Modo de entrenamiento
    optimizer = optim.AdamW(vae.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for videos, _ in dataloader:  # Solo usamos los videos en esta fase
            videos = videos.to(device)

            # Forward pass a través del VAE
            recon_videos, mu, logvar = vae(videos)

            # Calcular la pérdida VAE
            loss = vae_loss(recon_videos, videos, mu, logvar)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"VAE Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Guardar los pesos del VAE al terminar el entrenamiento
    torch.save(vae.state_dict(), 'vae_weights.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_file = '/content/data/video_descriptions.json'
video_dir = '/content/data/Videos/'
batch_size = 1
max_length = 512
#num_epochs = 20

transform = transforms.Compose([
    transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),  # Convertir cada frame a tensor
    transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)), # Switch to (C, T, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dimension: (1, C, T, H, W)
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, size=(16, 64, 64), mode='trilinear', align_corners=False)),  # Interpolación
    transforms.Lambda(lambda x: x.squeeze(0)),  # Remove batch dimension: (C, T, H, W)
    transforms.Lambda(lambda x: x * 2 - 1)  # Normalize to [-1, 1]
  ])

dataset = RealVideoDataset(json_file, video_dir, transform=transform, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instanciar el VAE
vae = VAE3D(in_channels=3, latent_dim=1024).to(device)

# Entrenar el VAE
train_vae_only(vae, dataloader, num_epochs=10, device=device)

vgg_model = VGGPerceptualLoss().to(device)


def train_full_model(model, dataloader, num_epochs, device, vgg_model):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for videos, texts in dataloader:
            videos, texts = videos.to(device), texts.to(device)
            
            print(f"Input video shape: {videos.shape}")
            print(f"Input text shape: {texts.shape}")

            # Forward pass a través del modelo completo
            recon_videos, mu, logvar = model(videos, texts)
            
            print(f"Reconstructed video shape: {recon_videos.shape}")
            print(f"mu shape: {mu.shape}")
            print(f"logvar shape: {logvar.shape}")

            # Calcular la pérdida total del modelo completo, incluyendo el modelo VGG
            loss = total_loss(recon_videos, videos, mu, logvar, vgg_model)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Full Model Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")



# Cargar los pesos entrenados del VAE
vae = VAE3D(in_channels=3, latent_dim=1024)
vae.load_state_dict(torch.load('vae_weights.pth'))  # Cargar los pesos guardados del VAE

# Instanciar el modelo Theseus con el VAE entrenado
transformer = ExpertTransformer(vocab_size=len(dataset.tokenizer), embed_dim=512)
model = Theseus(vae, transformer).to(device)

# Entrenar el modelo completo
train_full_model(model, dataloader, num_epochs=10, device=device, vgg_model=vgg_model)





def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for videos, texts in dataloader:
            videos = videos.to(device)
            texts = texts.to(device)

            reconstructed_videos, _, _ = model(videos, texts)

            if reconstructed_videos.size(2) != videos.size(2):
                reconstructed_videos = reconstructed_videos[:, :, :videos.size(2), :, :]

            loss = criterion(reconstructed_videos, videos)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


json_file = '/content/data/video_descriptions.json'
video_dir = '/content/data/Videos/'
batch_size = 1
max_length = 512
#num_epochs = 20

transform = transforms.Compose([
    transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),  # Convertir cada frame a tensor
    transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)), # Switch to (C, T, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dimension: (1, C, T, H, W)
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, size=(16, 64, 64), mode='trilinear', align_corners=False)),  # Interpolación
    transforms.Lambda(lambda x: x.squeeze(0)),  # Remove batch dimension: (C, T, H, W)
    transforms.Lambda(lambda x: x * 2 - 1)  # Normalize to [-1, 1]
  ])

dataset = RealVideoDataset(json_file, video_dir, transform=transform, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Cargar los pesos entrenados del VAE en Theseus
#vae = VAE3D(in_channels=3, latent_dim=1024)
#vae.load_state_dict(torch.load('vae_weights.pth'))  # Cargar los pesos guardados

#transformer = ExpertTransformer(vocab_size=len(dataset.tokenizer), embed_dim=512)
#model = Theseus(vae, transformer).to(device)

# Entrenar el modelo completo
#train_full_model(model, dataloader, num_epochs=100, device=device)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)

#train_model(model, dataloader, num_epochs, device)

avg_loss = evaluate_model(model, dataloader, device)
print(f"Average reconstruction loss: {avg_loss:.4f}")

# Guardar algunos videos reconstruidos
for i, (video, text) in enumerate(dataloader):
    if i >= 3:  # Limitar el número de videos reconstruidos a 3
        break

    video = video.to(device)
    text = text.to(device)

    # Reconstruir el video usando el modelo entrenado
    reconstructed_video, _, _ = model(video, text)

    # Guardar el video reconstruido
    save_video(reconstructed_video[0], f'reconstructed_video_{i+1}.mp4')

print("Evaluation complete and reconstructed videos saved.")

