# Import necessary modules
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import ViTImageProcessor, GPT2Tokenizer, ViTModel, GPT2LMHeadModel
from torch.optim import AdamW

# Initialize WandB
wandb.init(project="captioning_flickr30k", name="fine_tuning_with_validation")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
flickr30k = load_dataset("nlphuji/flickr30k", cache_dir="./new_cache_dir")
dataset = flickr30k["test"]  # Use the single available split

# Define split sizes
train_size = int(0.8 * len(dataset))  # 80% training
val_size = int(0.1 * len(dataset))    # 10% validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Ensure reproducibility
)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# Limit datasets to 500 examples for quick testing
train_dataset = Subset(train_dataset, range(min(500, len(train_dataset))))
val_dataset = Subset(val_dataset, range(min(500, len(val_dataset))))
test_dataset = Subset(test_dataset, range(min(500, len(test_dataset))))

# Initialize Tokenizer and Feature Extractor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use eos token as pad token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)  # Explicitly set pad_token_id

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
        return_tensors="pt"
    )
    return {
        "image": image,
        "input_ids": tokenized_caption["input_ids"][0],
        "attention_mask": tokenized_caption["attention_mask"][0],
        "caption": example["caption"]  # Include the actual caption for comparison
    }

# Preprocess datasets
def preprocess_dataset(subset, original_dataset):
    processed_data = []
    for idx in subset.indices:  # Use indices to access the original dataset
        example = original_dataset[idx]
        processed_data.append(preprocess(example))
    return processed_data

# Apply preprocessing
train_dataset = preprocess_dataset(train_dataset, dataset)
val_dataset = preprocess_dataset(val_dataset, dataset)
test_dataset = preprocess_dataset(test_dataset, dataset)

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
            "input_ids": item["input_ids"].clone().detach(),
            "attention_mask": item["attention_mask"].clone().detach(),
            "caption": item["caption"]
        }

# Create DataLoaders
train_loader = DataLoader(Flickr30kDataset(train_dataset), batch_size=8, shuffle=True)
val_loader = DataLoader(Flickr30kDataset(val_dataset), batch_size=8, shuffle=False)
test_loader = DataLoader(Flickr30kDataset(test_dataset), batch_size=8, shuffle=False)

# Encoder-Decoder Wrapper
class EncoderDecoderWrapper(nn.Module):
    def __init__(self, vit_model, gpt2_model, embed_dim):
        super().__init__()
        self.vit_model = vit_model
        self.gpt2_model = gpt2_model
        self.image_projector = nn.Linear(vit_model.config.hidden_size, embed_dim)
    
    def forward(self, image_inputs, caption_inputs):
        # Encode image
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

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Fine-Tuning Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        image_inputs = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        caption_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Forward pass
        outputs = model(image_inputs, caption_inputs)
        loss = outputs.loss
        total_train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log batch loss
        wandb.log({"Train Batch Loss": loss.item()})

    # Log epoch metrics
    avg_train_loss = total_train_loss / len(train_loader)
    wandb.log({"Epoch Train Loss": avg_train_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

    # Validation with WandB Logging for Images and Captions
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            image_inputs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            caption_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            # Forward pass
            outputs = model(image_inputs, caption_inputs)
            total_val_loss += outputs.loss.item()

            # Generate captions for WandB logging
            generated_ids = gpt2_model.generate(
                inputs_embeds=model.image_projector(vit_model(pixel_values=image_inputs).last_hidden_state[:, 0, :]).unsqueeze(1),
                max_length=20,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
            )
            generated_captions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

            # Log images and captions
            for img, actual_caption, generated_caption in zip(batch["image"], batch["caption"], generated_captions):
                wandb.log({
                    "Validation Example": wandb.Image(
                        img.permute(1, 2, 0).numpy(),  # Convert to HWC format
                        caption=f"Actual: {actual_caption}\nGenerated: {generated_caption}"
                    )
                })

    avg_val_loss = total_val_loss / len(val_loader)
    wandb.log({"Epoch Validation Loss": avg_val_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

# Save Model Components
vit_model.save_pretrained("./fine_tuned_captioning_model/vit_model")
gpt2_model.save_pretrained("./fine_tuned_captioning_model/gpt2_model")
torch.save(model.state_dict(), "./fine_tuned_captioning_model/encoder_decoder_wrapper.pth")
print("Model saved successfully!")
