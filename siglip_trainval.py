# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, SiglipModel, AutoProcessor
from PIL import Image
import requests

from io import BytesIO
import os
import torchvision.transforms as transforms
import json

class MemeDataset(Dataset):
    def __init__(self, image_dir="images/train", captions_file="memes-trainval.json", transform=None):
        self.image_dir = image_dir
        
        # Load the captions from the JSON file
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        # Create a mapping from img_fname to the corresponding item in captions_data
        self.captions_dict = {item['img_fname']: item for item in self.captions_data}

        # List all images in the directory that have a corresponding entry in the captions_dict
        self.image_files = [f for f in os.listdir(image_dir) if f in self.captions_dict]
        print(f"Total images found with captions: {len(self.image_files)}")  # For debugging

        # Use provided transform or a default transform if not specified
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a consistent size
            transforms.ToTensor()  # Converts the image to a tensor with values in [0, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_fname)

        # Open the image
        try:
            image = Image.open(img_path)

            # Convert the image to RGB format if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply the transformation
            image = self.transform(image)

            # Get the corresponding captions
            captions_info = self.captions_dict[img_fname]
            img_captions = " ".join(captions_info['img_captions']) if captions_info['img_captions'] else ""
            meme_captions = " ".join(captions_info['meme_captions']) if captions_info['meme_captions'] else ""

            return image, img_captions, meme_captions  # Return the image tensor and the captions
        except Exception as e:
            print(f"Error loading image {img_fname}: {e}")
            return None  # Return None if there is an error
        

import torch
from torch import nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(0.07))  # Learnable temperature
        self.b = nn.Parameter(torch.tensor(0.0))        # Learnable bias

    def forward(self, image_embeds, caption_embeds):
        # Step 1: Normalize the embeddings
        zimg = F.normalize(image_embeds, p=2, dim=1)
        ztxt = F.normalize(caption_embeds, p=2, dim=1)

        # Step 2: Compute logits
        t = torch.exp(self.t_prime)
        logits = zimg @ ztxt.T * t + self.b  # Pairwise similarity logits

        # Step 3: Construct labels
        n = image_embeds.size(0)  # Batch size
        labels = 2 * torch.eye(n, device=logits.device) - torch.ones((n, n), device=logits.device)

        # Step 4: Compute Sigmoid Loss
        loss = -torch.mean(F.logsigmoid(labels * logits))  # Average loss over all pairs
        return loss


# # Evaluation (Example Query)
# model.eval()
# query = "Spider Man making a suggestion"  # Example query
# inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
# with torch.no_grad():
#     query_embeds = model.get_text_features(**inputs)

# Now, you can use cosine similarity or nearest neighbors to retrieve top-k meme images



# %%
# %%
# Load the CLIP model and processor
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Call this before model initialization and training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")


# %%
# %%
# Load your dataset
import json
from torch.utils.data import DataLoader, random_split
# Load your dataset from a JSON file
with open('memes-trainval.json', 'r') as f:
    train_data = json.load(f)

from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Call this before model initialization and training

def collate_fn(batch):
    # Filter out empty entries
    batch = [item for item in batch if item]
    if len(batch) == 0:
        print("empty batch?")

    return default_collate(batch)

# Update your DataLoader
train_dataset = MemeDataset(image_dir="images/train", captions_file="memes-trainval.json")

test_dataset = MemeDataset(image_dir="images/test", captions_file="memes-test.json")

# Calculate the number of samples for training and validation
train_size = int(0.9 * len(train_dataset))  # 90% for training
val_size = len(train_dataset) - train_size  # Remaining 10% for validation

# Split the dataset
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Test samples: {len(test_dataset)}")

print(train_loader)

# %%

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def validate_model(model, processor, val_loader, device, top_k=5):
    model.eval()
    image_embeddings = []
    text_embeddings = []
    ground_truth = []

    with torch.no_grad():
        for images, img_captions, meme_captions in val_loader:  # Ignore meme_captions
            # Move images to the GPU
            images = images.to(device)

            # Generate embeddings for the images
            image_inputs = processor.image_processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            image_outputs = model.get_image_features(**image_inputs)
            image_embeddings.append(image_outputs.cpu().numpy())

            # Generate embeddings for the image captions
            img_captions = list(img_captions)
            img_caption_inputs = processor.tokenizer(img_captions, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            text_outputs = model.get_text_features(**img_caption_inputs)
            text_embeddings.append(text_outputs.cpu().numpy())

            # Append ground truth indices
            batch_size = len(img_captions)
            ground_truth.extend(range(len(ground_truth), len(ground_truth) + batch_size))

    # Convert lists to numpy arrays
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    print("Image Embeddings Shape:", image_embeddings.shape)
    print("Text Embeddings Shape:", text_embeddings.shape)

    # Calculate cosine similarities between text and image embeddings
    similarities = cosine_similarity(text_embeddings, image_embeddings)
    print("Similarities Shape:", similarities.shape)

    # Evaluate top-K accuracy
    top_k_accuracy = 0
    for idx, sim in enumerate(similarities):
        top_k_indices = np.argsort(sim)[-top_k:][::-1]  # Get top-K indices

        # Check if the ground truth index is in the top-K
        if ground_truth[idx] in top_k_indices:
            top_k_accuracy += 1

    # Calculate the percentage of correct top-K predictions
    top_k_accuracy /= len(ground_truth)
    print(f"Top-{top_k} Accuracy: {top_k_accuracy * 100:.2f}%")

    return top_k_accuracy

# Run validation
# val_accuracy = validate_model(model, processor, val_loader, device, top_k=5)


# %%
# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1} with validation loss: {loss:.4f}")

# %%
# Initialize the best validation loss as infinity
best_val_accuracy = 0

# %%
# Define optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8, weight_decay=1e-4)
contrastive_loss = SigLIPLoss()

# Fine-tuning loop
for epoch in range(50):  # Number of epochs
    model.train()
    total_loss = 0
    batch_count = 0
    for images, img_captions, meme_captions in train_loader:  # Ignore meme_captions
        # Move images to the GPU
        images = images.to(device)

        # Process images with img_captions
        img_captions = list(img_captions)
        img_caption_inputs = processor.tokenizer(img_captions, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        image_inputs = processor.image_processor(images=images, return_tensors="pt", do_rescale=False).to(device)

        # Get embeddings for images with img_captions
        inputs = {
            "pixel_values": image_inputs["pixel_values"].to(device),  # Ensure on GPU
            "input_ids": img_caption_inputs["input_ids"].to(device),  # Ensure on GPU
        }

        outputs = model(**inputs)
        img_caption_embeds = outputs.text_embeds

        # Compute contrastive loss for the image-caption pair
        loss = contrastive_loss(outputs.image_embeds, img_caption_embeds)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_count += 1
        # Print progress every 20 batches
        if batch_count % 20 == 0:
            progress = (batch_count * 8) / 5430 * 100  # Calculate progress percentage
            print(f"Epoch [{epoch+1}/50], Batch [{batch_count}], Progress: {progress:.2f}%")

    # Perform validation
    model.eval()
    val_accuracy = validate_model(model, processor, test_loader, device, top_k=5)

    # Save the best checkpoint if the validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint_path = f"best_checkpoint_meme_captions_new_last_epoch_new_siglip_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch, best_val_accuracy, checkpoint_path)

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


# %%
# %%
#test set
# %%
# Load the saved model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from '{checkpoint_path}' at epoch {checkpoint['epoch'] + 1}")
    return checkpoint['epoch'], checkpoint['loss']


# %%
# Path to the best checkpoint
checkpoint_path = "checkpoint_path"  # Replace <epoch_number> with the actual number

# Load the best checkpoint
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)  # Re-initialize optimizer for checkpoint loading
epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
print(best_loss)
# Evaluate on the test set
test_accuracy = validate_model(model, processor, test_loader, device, top_k=1)


