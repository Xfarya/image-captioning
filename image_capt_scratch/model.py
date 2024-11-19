import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from datasets import load_dataset
from PIL import Image
import wandb
import numpy as np
import math

# Define preprocess_pil_image
def preprocess_pil_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:  # Grayscale image
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    else:  # RGB image
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)
    return image_tensor


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim_size, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_size, 2).float() * (-math.log(10000.0) / dim_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim_size, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_size, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_size, dim_size * 4),
            nn.ReLU(),
            nn.Linear(dim_size * 4, dim_size),
        )
        self.norm1 = nn.LayerNorm(dim_size)
        self.norm2 = nn.LayerNorm(dim_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    def __init__(self, dim_size, num_heads, num_layers=3, dropout=0.1):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, dim_size, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )
        self.positional_encoding = PositionalEncoding(dim_size, max_len=28 * 28, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(dim_size, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        x = self.conv(x)
        batch_size, dim_size, H, W = x.shape
        x = x.view(batch_size, dim_size, H * W).transpose(1, 2)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_size, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim_size, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_size, dim_size * 4),
            nn.ReLU(),
            nn.Linear(dim_size * 4, dim_size),
        )
        self.norm1 = nn.LayerNorm(dim_size)
        self.norm2 = nn.LayerNorm(dim_size)
        self.norm3 = nn.LayerNorm(dim_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_size, num_heads, num_layers=3, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.positional_encoding = PositionalEncoding(dim_size, max_len=100, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(dim_size, num_heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(dim_size, vocab_size)

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        seq_len = x.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)
        logits = self.fc_out(x)
        return logits

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def generate_caption(image_tensor, encoder, decoder, tokenizer, device, max_length=16):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        encoder_output = encoder(image_tensor.unsqueeze(0).to(device))
        decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(device)
        generated_caption_ids = []
        for _ in range(max_length):
            logits = decoder(decoder_input_ids, encoder_output)
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1)
            generated_caption_ids.append(next_token_id.item())
            if next_token_id.item() == tokenizer.sep_token_id:
                break
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(0)], dim=1)
        tokens = tokenizer.convert_ids_to_tokens(generated_caption_ids, skip_special_tokens=True)
        return tokenizer.convert_tokens_to_string(tokens)


def train_on_10_examples(examples, encoder, decoder, tokenizer, device, epochs=50, learning_rate=0.001, log_every=10):
    wandb.init(project="caption-flickr-10", name="ten_image_training")
    encoder.to(device)
    decoder.to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(1, epochs + 1):
        total_loss = 0

        table = wandb.Table(columns=["Image", "Actual Caption", "Generated Caption"])
        encoder.train()
        decoder.train()

        for image, caption in examples:
            optimizer.zero_grad()

            image_tensor = preprocess_pil_image(image).to(device).unsqueeze(0)
            encoder_output = encoder(image_tensor)

            input_ids = tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=16)["input_ids"].to(device)
            decoder_input_ids = input_ids.clone()
            decoder_input_ids[:, 1:] = input_ids[:, :-1]
            decoder_input_ids[:, 0] = tokenizer.cls_token_id

            logits = decoder(decoder_input_ids, encoder_output)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch % log_every == 0 or epoch == 1:
                encoder.eval()
                decoder.eval()
                generated_caption = generate_caption(image_tensor.squeeze(0), encoder, decoder, tokenizer, device)
                img = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                table.add_data(wandb.Image(img), caption, generated_caption)

        wandb.log({"epoch": epoch, "loss": total_loss / len(examples), "Image and Captions": table})

    wandb.finish()


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(dim_size=128, num_heads=8, num_layers=4)
    decoder = Decoder(vocab_size=tokenizer.vocab_size, dim_size=128, num_heads=8, num_layers=4)

    dataset = load_dataset("nlphuji/flickr30k", split="test", streaming=True)
    examples = [(example["image"], example["caption"][0]) for _, example in zip(range(10), iter(dataset))]

    train_on_10_examples(
        examples=examples,
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=50,
        learning_rate=0.001,
        log_every=10
    )
