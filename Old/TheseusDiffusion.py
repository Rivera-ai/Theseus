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


class DiffusionEncoder(nn.Module):
    def __init__(self, timesteps, latent_dim=1024):
        super(DiffusionEncoder, self).__init__()
        self.timesteps = timesteps
        self.diffusion_steps = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(timesteps)
        ])
    
    def forward(self, z):
        for t in range(self.timesteps):
            noise = torch.randn_like(z)
            z = z + noise  # Añadir ruido
            z = self.diffusion_steps[t](z)  # Predecir el próximo paso
        return z

class DiffusionDecoder(nn.Module):
    def __init__(self, timesteps, latent_dim=1024):
        super(DiffusionDecoder, self).__init__()
        self.timesteps = timesteps
        self.diffusion_steps = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(timesteps)
        ])
    
    def forward(self, z):
        for t in reversed(range(self.timesteps)):
            z = self.diffusion_steps[t](z)
        return z

class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, timesteps=1000):
        super(DiffusionTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.diffusion_steps = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(timesteps)])
    
    def forward(self, x_text):
        x = self.embedding(x_text)
        x = self.transformer(x)
        for step in self.diffusion_steps:
            x = step(x)
        return x.mean(dim=1)



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
    def __init__(self, vae, diffusion_encoder, diffusion_decoder, transformer):
        super(Theseus, self).__init__()
        self.vae = vae
        self.diffusion_encoder = diffusion_encoder
        self.diffusion_decoder = diffusion_decoder
        self.transformer = transformer
        self.text_mapper = nn.Linear(512, 1024)

    def forward(self, x_video, x_text):
        # 1. Comprimir el video usando el VAE para obtener el espacio latente del video
        mu, logvar = self.vae.encode(x_video)
        z_vision = self.vae.reparameterize(mu, logvar)

        # 2. Procesar el texto con el transformer para obtener la representación textual
        z_text = self.transformer(x_text)
        z_text = self.text_mapper(z_text)

        # 3. Aplicar el encoder de difusión para añadir ruido al espacio latente del video
        z_noisy = self.diffusion_encoder(z_vision)

        # 4. Combinar la representación de texto y el espacio latente ruidoso del video
        z_combined = z_noisy + z_text

        # 5. Aplicar el decoder de difusión para reconstruir el video deshaciendo el ruido
        reconstructed_latent = self.diffusion_decoder(z_combined)

        # 6. Decodificar el espacio latente para obtener el video reconstruido
        reconstructed_video = self.vae.decode(reconstructed_latent)

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

def diffusion_loss(x_pred, x_real):
    return F.mse_loss(x_pred, x_real)

def total_loss(recon_x, x, mu, logvar, diffusion_loss_weight=0.1):
    vae_loss_value = vae_loss(recon_x, x, mu, logvar)
    diffusion_loss_value = diffusion_loss(recon_x, x)
    return vae_loss_value + diffusion_loss_weight * diffusion_loss_value



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

json_file = '/teamspace/studios/this_studio/data/video_descriptions.json'
video_dir = '/teamspace/studios/this_studio/data/Videos'
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
train_vae_only(vae, dataloader, num_epochs=50, device=device)

vgg_model = VGGPerceptualLoss().to(device)


def train_full_model(model, dataloader, num_epochs, device, vgg_model):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for videos, texts in dataloader:
            videos, texts = videos.to(device), texts.to(device)

            # Forward pass a través del modelo completo (incluyendo difusión)
            recon_videos, mu, logvar = model(videos, texts)
            

            # Calcular la pérdida total
            vae_loss_value = vae_loss(recon_videos, videos, mu, logvar)
            diffusion_loss_value = diffusion_loss(recon_videos, videos)
            perceptual_loss_value = perceptual_loss(recon_videos, videos, vgg_model)
            loss = vae_loss_value + 0.1 * diffusion_loss_value + perceptual_loss_value
            
            total_vae_loss += vae_loss_value.item()
            total_diffusion_loss += diffusion_loss_value.item()
            total_perceptual_loss += perceptual_loss_value.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, VAE Loss: {total_vae_loss/len(dataloader):.4f}, "
              f"Diffusion Loss: {total_diffusion_loss/len(dataloader):.4f}, "
              f"Perceptual Loss: {total_perceptual_loss/len(dataloader):.4f}")




vae = VAE3D(in_channels=3, latent_dim=1024)
vae.load_state_dict(torch.load('vae_weights.pth'))  

# Instanciar el DiffusionEncoder, DiffusionDecoder y el DiffusionTransformer
diffusion_encoder = DiffusionEncoder(timesteps=100, latent_dim=1024).to(device)
diffusion_decoder = DiffusionDecoder(timesteps=100, latent_dim=1024).to(device)
transformer = DiffusionTransformer(vocab_size=len(dataset.tokenizer), embed_dim=512).to(device)

# Instanciar el modelo completo Theseus
model = Theseus(vae, diffusion_encoder, diffusion_decoder, transformer).to(device)

# Entrenar el modelo completo
num_epochs_full = 50
train_full_model(model, dataloader, num_epochs=num_epochs_full, device=device, vgg_model=vgg_model)



def evaluate_model(model, dataloader, device):
    model.eval()  # Poner el modelo en modo evaluación
    total_loss = 0
    criterion = nn.MSELoss()  # Usar MSE para la pérdida de reconstrucción

    with torch.no_grad():  # Desactivar el cálculo de gradientes para acelerar la inferencia
        for videos, texts in dataloader:
            videos = videos.to(device)
            texts = texts.to(device)

            # Forward pass a través del modelo completo (incluyendo difusión y transformer)
            reconstructed_videos, mu, logvar = model(videos, texts)

            # Asegurar que las dimensiones del video reconstruido coincidan con el video original
            if reconstructed_videos.size(2) != videos.size(2):
                reconstructed_videos = reconstructed_videos[:, :, :videos.size(2), :, :]

            # Calcular la pérdida de reconstrucción entre el video original y el reconstruido
            loss = criterion(reconstructed_videos, videos)
            total_loss += loss.item()

    # Calcular la pérdida promedio en todo el dataset
    avg_loss = total_loss / len(dataloader)
    return avg_loss



json_file = '/teamspace/studios/this_studio/data/video_descriptions.json'
video_dir = '/teamspace/studios/this_studio/data/Videos'
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

# Configurar el dataloader y el modelo para la evaluación
dataset = RealVideoDataset(json_file, video_dir, transform=transform, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Cargar los pesos entrenados del VAE y configurar el modelo
vae = VAE3D(in_channels=3, latent_dim=1024)
vae.load_state_dict(torch.load('vae_weights.pth'))  # Cargar pesos del VAE

# Instanciar el DiffusionEncoder, DiffusionDecoder y el DiffusionTransformer
diffusion_encoder = DiffusionEncoder(timesteps=100, latent_dim=1024).to(device)
diffusion_decoder = DiffusionDecoder(timesteps=100, latent_dim=1024).to(device)
transformer = DiffusionTransformer(vocab_size=len(dataset.tokenizer), embed_dim=512).to(device)

# Instanciar el modelo completo Theseus
model = Theseus(vae, diffusion_encoder, diffusion_decoder, transformer).to(device)

# Evaluar el modelo
avg_loss = evaluate_model(model, dataloader, device)
print(f"Average reconstruction loss: {avg_loss:.4f}")

# Guardar algunos videos reconstruidos para visualización
for i, (video, text) in enumerate(dataloader):
    if i >= 3:  # Limitar el número de videos reconstruidos a 3
        break

    video = video.to(device)
    text = text.to(device)

    # Generar el video reconstruido
    reconstructed_video, _, _ = model(video, text)

    # Guardar el video reconstruido en formato MP4
    save_video(reconstructed_video[0], f'reconstructed_video_{i+1}.mp4')

print("Evaluation complete and reconstructed videos saved.")
