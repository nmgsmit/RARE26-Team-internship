import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from sklearn.model_selection import train_test_split
import wandb
 
# Dataset structure:
# data/
# ├── center_1/
# │   ├── ndbe/  
# │   └── neo/  
# └── center_2/
#     ├── ndbe/
#     └── neo/ 
### 

def get_args_parser():
    parser = ArgumentParser("RARE25 Classification Training")
    # Change the default path to match your folder name!
    parser.add_argument("--data-dir", type=str, default="./data", help="Where you put center_1, center_2, etc.")
    parser.add_argument("--DatasetSplit", type=int, default=80, help="Percentage of images for training (rest for validation)")
    parser.add_argument("--batch-size", type=int, default=32, help="How many images to look at once")
    parser.add_argument("--epochs", type=int, default=20, help="How many times to loop over the whole dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="How fast the model 'learns'")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU power for loading images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="rare25-test-run")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Where to save the trained model")
    return parser

def main(args):
    # Log into Weights & Biases so we can see the graphs later
    wandb.init(project="RARE25-Project", name=args.experiment_id, config=vars(args))
    
    # Setup directories and devices
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save directory exists
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATA ---------------------------------------------------------------------------------------------------------------
    # In the coming part different steps of data loading and preparation are performed.
    # -------------------------------------------------------------------------------------------------------- DATA LOADING 
    # Check if the data is in a folder, if yes then this step will be skipped. Otherwise we download via huggingface.
    # Do note that you need to fill in your personal huggingface token in the .env file for this to work.

    if not os.path.exists(args.data_dir):
        from huggingface_hub import snapshot_download 
        from dotenv import load_dotenv, find_dotenv
        
        print("Data not found locally. Downloading folders from Hugging Face...")
        
        load_dotenv(find_dotenv())
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("Could not find HF_TOKEN in .env file! Make sure it is set.")

        # Download the repo contents directly into your ./data folder!
        snapshot_download(
            repo_id="TimJaspersTue/RARE25-train", 
            repo_type="dataset",
            local_dir=args.data_dir, 
            token=hf_token      
        )
        print("Data downloaded successfully.")
    # ----------------------------------------------------------------------------------------------------- DATA PREPARATION 
    # Standard: resize to 224x224 (quite standard), we can change it later!
    transform = Compose([
        ToImage(),
        Resize((224, 224)), 
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization of the colour channels (standard)
    ])

    # <<!-- We can add more transformations here if you want to experiment with data augmentation. -->

    # --------------------------------------------------------------------------------------------- COMBINE DATASETS AND SPLIT
    # ImageFolder will automatically assign labels based on folder names (ndbe=0, neo=1).
    centers = [f for f in os.listdir(args.data_dir) if f.startswith('center')]
    train_datasets = []
    valid_datasets = []
    
    for center in centers:
        center_path = os.path.join(args.data_dir, center)
        ds = ImageFolder(root=center_path, transform=transform)
            
        # SAFETY CHECK: Ensure labels are consistently 0=ndbe, 1=neo across all centers!
        assert ds.class_to_idx == {'ndbe': 0, 'neo': 1}, f"CRITICAL WARNING: Class mapping in {center} is backwards or broken: {ds.class_to_idx}"

        # STRATIFIED SPLIT: Split this specific center while maintaining ndbe/neo ratios
        # We extract ds.targets (the labels) to tell sklearn how to balance the split
        train_idx, val_idx = train_test_split(
            range(len(ds)), 
            train_size=args.DatasetSplit / 100.0, 
            stratify=ds.targets, 
            random_state=args.seed
            )
            
        # Append the subsets to our lists
        train_datasets.append(Subset(ds, train_idx))
        valid_datasets.append(Subset(ds, val_idx))
    
    # Merge all the center subsets together
    train_ds = ConcatDataset(train_datasets)
    valid_ds = ConcatDataset(valid_datasets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # >>>>>>>>>> FROM HERE IT SHOULD BE MODIFIED FOR THE MODEL IMPLEMENTATION, TRAINING AND EVALUATION. <<<<<<<<<<
    # # MODEL SETUP ----------------------------------------------------------------------------------------------------------
    # from model import Model # Uses the model in model.py. 
    # model = Model(n_classes=2).to(device) # We only have 2 classes: ndbe and neo.
    
    # # CrossEntropyLoss is the standard for classification.
    # criterion = nn.CrossEntropyLoss()
    # optimizer = AdamW(model.parameters(), lr=args.lr)

    # # ------------------------------------------------------------------------------------------------------- TRAINING LOOP
    # best_valid_loss = float('inf')

    # for epoch in range(args.epochs):
    #     # TRAINING
    #     model.train()
    #     epoch_loss = 0
        
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
            
    #         optimizer.zero_grad()      # Reset math from last step
    #         outputs = model(images)     # Guess what the image is
    #         loss = criterion(outputs, labels) # See how wrong the guess was
    #         loss.backward()            # Calculate how to improve
    #         optimizer.step()           # Actually improve the weights
            
    #         epoch_loss += loss.item()
    #     avg_train_loss = epoch_loss / len(train_loader)

    #     # VALIDATION 
    #     model.eval()
    #     valid_loss = 0
    #     correct_predictions = 0
    #     total_predictions = 0
    #     with torch.no_grad():
    #         for images, labels in valid_loader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
    #             valid_loss += loss.item()
                
    #             # Calculate accuracy
    #             _, predicted = torch.max(outputs.data, 1)
    #             total_predictions += labels.size(0)
    #             correct_predictions += (predicted == labels).sum().item()
    #     avg_valid_loss = valid_loss / len(valid_loader)
    #     valid_accuracy = correct_predictions / total_predictions

    #     # Log our progress to WandB
    #     wandb.log({
    #         "train_loss": avg_train_loss,
    #         "valid_loss": avg_valid_loss,
    #         "valid_accuracy": valid_accuracy,
    #         "epoch": epoch + 1
    #     })
        
    #     print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_valid_loss:.4f} | Val Acc: {valid_accuracy:.4f}")

    #     # MODEL SAVING
    #     # If the model improved on the validation set, save it!
    #     if avg_valid_loss < best_valid_loss:
    #         best_valid_loss = avg_valid_loss
    #         save_path = os.path.join(args.save_dir, f"{args.experiment_id}_best.pt")
    #         torch.save(model.state_dict(), save_path)
    #         print(f"   -> Saved new best model to {save_path}")

    # # Save final model state after all epochs finish
    # final_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_final.pt")
    # torch.save(model.state_dict(), final_save_path)

    # print("Training finished! Check your WandB dashboard.")
    # wandb.finish()

if __name__ == "__main__":
    main(get_args_parser().parse_args())
