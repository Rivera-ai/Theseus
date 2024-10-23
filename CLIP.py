import torch
import torch.nn as nn
import transformers
from transformers import CLIPTextModel, CLIPTokenizer

transformers.logging.set_verbosity_error()

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):

    def __init__(self, path="helenai/runwayml-stable-diffusion-v1-5-ov", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
        self.transformer = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")

        self.device = device
        self.max_length = max_length
        self._freeze()

    def _freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        )

        tokens = batch_encoding["inputs_ids"].to(self.device)
        mask = batch_encoding['attention_mask']
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
        return z, pooled_z, mask

class CLIPEncoder:
    def __init__(self, from_pretrained, model_max_length=77, device="cuda", dtype=torch.float):
        super().__init__()

        assert from_pretrained is not None, "Please specify the path to the T5 model"

        self.text_encoder = FrozenCLIPEmbedder(path=from_pretrained, max_length=model_max_length).to(device, dtype)

        in_channels = 768
        self.y_embedding = self.encode([""])['y'].squeeze()

        self.model_max_length = model_max_length
        self.output_dim = self.text_encoder.transformer.config.hidde_size

    def encode(self, text):
        embeddings, pooled_embeddings, mask = self.text_encoder.encode(text)
        y = embeddings.unsqueeze(1)
        return dict(y=y, mask=mask)

    def null(self, n):
        null_y = self.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y

    def to(self, dtype):
        self.text_encoder = self.text_encoder.to(dtype)
        return self
