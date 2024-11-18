import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import ViTFeatureExtractor, GPT2Tokenizer, ViTModel, GPT2LMHeadModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Flickr30k Dataset
flickr30k = load_dataset("nlphuji/flickr30k", cache_dir="./new_cache_dir")

# Combine shards into a single dataset and split it
test_dataset = flickr30k["test"].flatten_indices()
train_test_split = test_dataset.train_test_split(test_size=0.2)
validation_test_split = train_test_split["test"].train_test_split(test_size=0.5)

# Final splits
train_dataset = train_test_split["train"].select(range(10))  # Limit to 10 examples for quick testing
validation_dataset = validation_test_split["train"].select(range(10))
test_dataset = validation_test_split["test"].select(range(10))

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(validation_dataset)}")
print(f"Test examples: {len(test_dataset)}")

# Initialize Tokenizer and Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use eos token as pad token

# Define image transformations
transform = Compose([
    Resize((224, 224)),  # Resize all images to 224x224
    ToTensor()           # Convert images to PyTorch tensors
])

# Preprocessing Function for Images and Captions
def preprocess(example):
    # Resize and transform the image
    image = transform(example["image"])
    
    # Tokenize the caption
    tokenized_caption = tokenizer(
        example["caption"],
        padding="max_length",
        truncation=True,
        max_length=30,
        return_tensors="np"
    )
    return {
        "image": image,
        "input_ids": tokenized_caption["input_ids"][0],
        "attention_mask": tokenized_caption["attention_mask"][0],
    }

# Apply preprocessing to datasets
train_dataset = train_dataset.map(preprocess)
validation_dataset = validation_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Update dataset format for PyTorch
train_dataset.set_format(type="torch", columns=["image", "input_ids", "attention_mask"])
validation_dataset.set_format(type="torch", columns=["image", "input_ids", "attention_mask"])
test_dataset.set_format(type="torch", columns=["image", "input_ids", "attention_mask"])

# PyTorch Dataset Class
class Flickr30kDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": item["image"],
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

# Create DataLoaders
train_loader = DataLoader(Flickr30kDataset(train_dataset), batch_size=8, shuffle=True)
validation_loader = DataLoader(Flickr30kDataset(validation_dataset), batch_size=8)
test_loader = DataLoader(Flickr30kDataset(test_dataset), batch_size=8)

# Encoder-Decoder Wrapper
class EncoderDecoderWrapper(nn.Module):
    def __init__(self, vit_model, gpt2_model, embed_dim):
        super().__init__()
        self.vit_model = vit_model
        self.gpt2_model = gpt2_model
        self.image_projector = nn.Linear(vit_model.config.hidden_size, embed_dim)
    
    def forward(self, image_inputs, caption_inputs):
        # Encode image
        with torch.no_grad():
            image_features = self.vit_model(pixel_values=image_inputs).last_hidden_state
        
        # Project image features to GPT-2 embedding space
        projected_features = self.image_projector(image_features[:, 0, :])  # CLS token
        
        # Add image embeddings to caption embeddings
        caption_embeddings = self.gpt2_model.transformer.wte(caption_inputs["input_ids"])
        combined_embeddings = torch.cat([projected_features.unsqueeze(1), caption_embeddings], dim=1)
        
        # Adjust attention mask
        attention_mask = caption_inputs["attention_mask"]
        extended_attention_mask = torch.cat(
            [torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask], dim=1
        )
        
        # Adjust labels for loss computation
        labels = caption_inputs["input_ids"]
        extended_labels = torch.cat(
            [torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype), labels], dim=1
        )
        
        # Pass combined embeddings through GPT-2
        outputs = self.gpt2_model(
            inputs_embeds=combined_embeddings,
            labels=extended_labels,
            attention_mask=extended_attention_mask
        )
        return outputs

# Load Models
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize Model
model = EncoderDecoderWrapper(vit_model, gpt2_model, embed_dim=gpt2_model.config.n_embd).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(2):  # Reduce epochs for quick testing
    model.train()
    running_loss = 0
    for batch in train_loader:
        image_inputs = batch["image"].to(device)
        caption_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

        outputs = model(image_inputs, caption_inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}")

# Test Caption Generation
test_image_path = "images/Karim.jpeg"  # Replace with an actual image path
test_image = Image.open(test_image_path).convert("RGB")

# Preprocess the test image
inputs = feature_extractor(images=test_image, return_tensors="pt").to(device)

with torch.no_grad():
    # Extract image features
    image_features = vit_model(pixel_values=inputs["pixel_values"]).last_hidden_state
    projected_features = model.image_projector(image_features[:, 0, :]).unsqueeze(1)

    # Generate caption with attention mask
    attention_mask = torch.ones((projected_features.size(0), projected_features.size(1)), dtype=torch.long).to(device)
    generated_ids = gpt2_model.generate(
        inputs_embeds=projected_features,
        attention_mask=attention_mask,  # Explicitly set attention_mask
        max_length=20,
        eos_token_id=gpt2_model.config.eos_token_id,
        pad_token_id=gpt2_model.config.eos_token_id,  # Set pad_token_id explicitly
        do_sample=True,
        temperature=0.7
    )

    # Decode the generated caption
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated Caption:", caption)
