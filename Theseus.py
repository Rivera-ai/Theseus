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
            CausalConv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.fc_mu = nn.Linear(32768, latent_dim)
        self.fc_logvar = nn.Linear(32768, latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 512 * 1 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, in_channels, kernel_size=(3, 4, 4), stride=2, padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        print(h.shape)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(h.size(0), 512, 1, 4, 4)
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
        z_combined = z_vision + z_text.unsqueeze(1)  # Adjust for 3D tensors

        # Rebuild the video
        reconstructed_video = self.vae.decode(z_combined)

        return reconstructed_video, mu, logvar


def save_video(tensor, path):
    tensor = tensor.detach().permute(1, 2, 3, 0).cpu().numpy()  # Switch to (T, H, W, C)
    tensor = (tensor * 255).astype(np.uint8)  # Scalar from [0, 1] to [0, 255]

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (tensor.shape[2], tensor.shape[1]))

    for frame in tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def train_model(model, dataloader, num_epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (videos, texts) in enumerate(dataloader):
            videos = videos.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            reconstructed_videos, mu, logvar = model(videos, texts)

            # Recorta el video reconstruido a la longitud del video original
            if reconstructed_videos.size(2) != videos.size(2):
                reconstructed_videos = reconstructed_videos[:, :, :videos.size(2), :, :]

            reconstruction_loss = criterion(reconstructed_videos, videos)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + 0.001 * kl_divergence

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training complete")



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
num_epochs = 200

transform = transforms.Compose([
    transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames])),  # Convertir cada frame a tensor
    transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)), # Switch to (C, T, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dimension: (1, C, T, H, W)
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, size=(16, 64, 64), mode='trilinear', align_corners=False)),  # Interpolación
    transforms.Lambda(lambda x: x.squeeze(0))  # Remove batch dimension: (C, T, H, W)
])

dataset = RealVideoDataset(json_file, video_dir, transform=transform, max_length=512)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

vae = VAE3D(in_channels=3, latent_dim=1024)
transformer = ExpertTransformer(vocab_size=len(dataset.tokenizer), embed_dim=512)
model = Theseus(vae, transformer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(model, dataloader, num_epochs, device)

avg_loss = evaluate_model(model, dataloader, device)
print(f"Average reconstruction loss: {avg_loss:.4f}")

for i, (video, text) in enumerate(dataloader):
    if i >= 3:
        break

    video = video.to(device)
    text = text.to(device)

    reconstructed_video, _, _ = model(video, text)

    save_video(reconstructed_video[0], f'reconstructed_video_{i+1}.mp4')

print("Evaluation complete and reconstructed videos saved.")
