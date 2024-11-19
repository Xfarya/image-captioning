# Import necessary modules
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import ViTImageProcessor, GPT2Tokenizer, ViTModel, GPT2LMHeadModel
from torch.optim import AdamW

# Initialize WandB
wandb.init(project="captioning_flickr30k", name="fine_tuning_test_run")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and subset the dataset
flickr30k = load_dataset("nlphuji/flickr30k", cache_dir="./new_cache_dir")
subset_size = 100  # Updated from 500 to 10,000
full_dataset = flickr30k["test"]  # Only "test" split is available

# Define split sizes
train_size = int(0.8 * subset_size)  # 8,000
val_size = int(0.1 * subset_size)    # 1,000
test_size = subset_size - train_size - val_size  # 1,000

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset.select(range(subset_size)),
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# Initialize Tokenizer and Feature Extractor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# Define image transformations
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

def preprocess(examples):
    captions = [caption[0] if isinstance(caption, list) else caption for caption in examples["caption"]]
    images = [transform(image) for image in examples["image"]]
    tokenized_captions = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=30,
        return_tensors="pt"
    )
    return {
        "image": torch.stack(images),
        "input_ids": tokenized_captions["input_ids"].tolist(),  # Convert to list for compatibility
        "attention_mask": tokenized_captions["attention_mask"].tolist(),  # Convert to list for compatibility
        "caption": captions
    }



# Apply preprocessing
def preprocess_dataset(dataset):
    return dataset.map(preprocess, batched=True, batch_size=32)

train_dataset = preprocess_dataset(train_dataset.dataset)
val_dataset = preprocess_dataset(val_dataset.dataset)
test_dataset = preprocess_dataset(test_dataset.dataset)

# PyTorch Dataset Class
class Flickr30kDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": torch.tensor(item["image"], dtype=torch.float32),  # Convert image to tensor
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),  # Convert input_ids to tensor
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),  # Convert attention_mask to tensor
            "caption": item["caption"]  # Keep the caption as is
        }


# Create DataLoaders
train_loader = DataLoader(Flickr30kDataset(train_dataset), batch_size=32, shuffle=True)
val_loader = DataLoader(Flickr30kDataset(val_dataset), batch_size=32, shuffle=False)
test_loader = DataLoader(Flickr30kDataset(test_dataset), batch_size=32, shuffle=False)

# Encoder-Decoder Wrapper
class EncoderDecoderWrapper(nn.Module):
    def __init__(self, vit_model, gpt2_model, embed_dim):
        super().__init__()
        self.vit_model = vit_model
        self.gpt2_model = gpt2_model
        self.image_projector = nn.Linear(vit_model.config.hidden_size, embed_dim)

    def forward(self, image_inputs, caption_inputs):
        image_features = self.vit_model(pixel_values=image_inputs).last_hidden_state
        projected_features = self.image_projector(image_features[:, 0, :])  # CLS token
        caption_embeddings = self.gpt2_model.transformer.wte(caption_inputs["input_ids"])
        combined_embeddings = torch.cat([projected_features.unsqueeze(1), caption_embeddings], dim=1)
        attention_mask = caption_inputs["attention_mask"]
        extended_attention_mask = torch.cat(
            [torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask], dim=1
        )
        labels = caption_inputs["input_ids"]
        extended_labels = torch.cat(
            [torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype), labels], dim=1
        )
        outputs = self.gpt2_model(
            inputs_embeds=combined_embeddings,
            labels=extended_labels,
            attention_mask=extended_attention_mask
        )
        return outputs

# Load Models
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Initialize Model
model = EncoderDecoderWrapper(vit_model, gpt2_model, embed_dim=gpt2_model.config.n_embd).to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Fine-Tuning Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        image_inputs = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        caption_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = model(image_inputs, caption_inputs)
        loss = outputs.loss
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"Train Batch Loss": loss.item()})

    avg_train_loss = total_train_loss / len(train_loader)
    wandb.log({"Epoch Train Loss": avg_train_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

    # Validation
    model.eval()
    total_val_loss = 0
    embeddings = []  # Store embeddings

    with torch.no_grad():
        for batch in val_loader:
            image_inputs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            caption_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            outputs = model(image_inputs, caption_inputs)
            total_val_loss += outputs.loss.item()
            # Save embeddings
            image_features = model.vit_model(pixel_values=image_inputs).last_hidden_state[:, 0, :].cpu()
            embeddings.append(image_features)

    avg_val_loss = total_val_loss / len(val_loader)
    wandb.log({"Epoch Validation Loss": avg_val_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

    # Concatenate all embeddings and save
    embeddings_tensor = torch.cat(embeddings, dim=0)
    torch.save(embeddings_tensor, f"./embeddings_epoch_{epoch + 1}.pth")

# Save Model
model_save_path = "./fine_tuned_captioning_model"
vit_model.save_pretrained(f"{model_save_path}/vit_model")
gpt2_model.save_pretrained(f"{model_save_path}/gpt2_model")
torch.save(model.state_dict(), f"{model_save_path}/encoder_decoder_wrapper.pth")
print("Model and embeddings saved successfully!")
