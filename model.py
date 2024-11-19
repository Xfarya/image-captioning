import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import ViTImageProcessor, GPT2Tokenizer, ViTModel, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Initialize WandB
wandb.init(project="captioning_flickr30k", name="test_run_with_metrics_and_table")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Flickr30k Dataset
flickr30k = load_dataset("nlphuji/flickr30k", cache_dir="./new_cache_dir")

# Combine shards into a single dataset and split it
test_dataset = flickr30k["test"].flatten_indices()

# Prepare a subset of 500 images for testing
test_dataset = test_dataset.select(range(500))

print(f"Test examples: {len(test_dataset)}")

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
        return_tensors="np"
    )
    
    attention_mask = tokenized_caption['attention_mask']
    
    return {
        "image": image,
        "input_ids": tokenized_caption["input_ids"][0],
        "attention_mask": attention_mask[0],  # Explicitly pass the attention mask
        "caption": example["caption"]  # Include the actual caption for comparison
    }

# Apply preprocessing to test dataset
test_dataset = test_dataset.map(preprocess)

# Update dataset format for PyTorch
test_dataset.set_format(type="torch", columns=["image", "input_ids", "attention_mask", "caption"])

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
            "caption": item["caption"]  # Pass the actual caption
        }

# Create DataLoader for testing
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
        with torch.no_grad():
            image_features = vit_model(image_inputs).last_hidden_state
        
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

# Initialize WandB Table
wandb_table = wandb.Table(columns=["Image", "Actual Caption", "Generated Caption"])

# Metrics
total_loss = 0
total_bleu_score = 0
total_correct = 0
total_tokens = 0
num_captions = 0

# Test Loop for Generating Captions
# Test Loop for Generating Captions
model.eval()
smooth_fn = SmoothingFunction().method1  # BLEU smoothing
for batch in test_loader:
    image_inputs = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    caption_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    with torch.no_grad():
        # Forward pass
        outputs = model(image_inputs, caption_inputs)
        loss = outputs.loss
        total_loss += loss.item()

        # Token-Level Accuracy Calculation
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)  # Predicted token IDs
        correct = (predictions[:, 1:] == input_ids).float() * attention_mask  # Ignore padding
        total_correct += correct.sum().item()
        total_tokens += attention_mask.sum().item()

        # Generate captions
        image_features = vit_model(image_inputs).last_hidden_state  # Corrected usage
        projected_features = model.image_projector(image_features[:, 0, :]).unsqueeze(1)
        generated_ids = gpt2_model.generate(
            inputs_embeds=projected_features,
            max_length=20,
            eos_token_id=gpt2_model.config.eos_token_id,
            pad_token_id=gpt2_model.config.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        generated_captions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

        # Compute BLEU for each caption
        actual_captions = batch["caption"]
        for actual, generated in zip(actual_captions, generated_captions):
            if isinstance(actual, tuple):  # Handle tuple captions
                actual = actual[0]  # Use the first caption as reference
            reference = [actual.split()]  # Tokenize the actual caption
            candidate = generated.split()  # Tokenize the generated caption
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
            total_bleu_score += bleu_score
            num_captions += 1

        # Log batch loss and accuracy to WandB
        batch_accuracy = correct.sum().item() / attention_mask.sum().item()
        wandb.log({"Batch Loss": loss.item(), "Batch Accuracy": batch_accuracy})

        # Add data to WandB table
        for img, actual_caption, generated_caption in zip(batch["image"], actual_captions, generated_captions):
            wandb_table.add_data(
                wandb.Image(img.permute(1, 2, 0).numpy()),  # Convert to HWC format
                actual_caption if not isinstance(actual_caption, tuple) else actual_caption[0],
                generated_caption
            )

# Calculate final metrics
avg_loss = total_loss / len(test_loader)
avg_bleu_score = total_bleu_score / num_captions
accuracy = total_correct / total_tokens  # Overall accuracy

# Log final metrics to WandB
wandb.log({
    "Average Test Loss": avg_loss,
    "Average BLEU Score": avg_bleu_score,
    "Token-Level Accuracy": accuracy,
    "Comparison Table": wandb_table
})
print(f"Average Test Loss: {avg_loss}")
print(f"Average BLEU Score: {avg_bleu_score}")
print(f"Token-Level Accuracy: {accuracy}")
