import os
import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

# Add environment variable settings for RTX 4000 series GPUs
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

class LatexImageDataset(Dataset):
    def __init__(self, mapping_file, image_dir, tokenizer, max_length=77):
        self.equations = []
        self.image_paths = []
        
        # Load the mapping between equations and their images
        with open(mapping_file, 'r') as f:
            for line in f:
                idx, equation = line.strip().split('||')
                self.equations.append(equation)
                self.image_paths.append(os.path.join(image_dir, f"equation_{idx}.png"))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.equations)
    
    def __getitem__(self, idx):
        equation = self.equations[idx]
        image_path = self.image_paths[idx]
        
        # Tokenize the LaTeX equation
        inputs = self.tokenizer(
            equation,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Load and transform the image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.image_transforms(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default black image if there's an error
            image = torch.zeros(3, 512, 512)
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "pixel_values": image
        }

def train(args):
    # Initialize accelerator with appropriate settings
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.with_tracking else None,
    )
    
    if accelerator.is_main_process:
        print(f"Device: {accelerator.device}")
        print(f"Using NCCL with P2P and IB disabled for RTX 4000 series compatibility")
    
    # Set up logging directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer, UNet, and noise scheduler
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    vae.requires_grad_(False)  # Freeze VAE parameters
    text_encoder.requires_grad_(False)
    
    # Freeze vae and text_encoder
    # We'll use the UNet from pretrained model but fine-tune it on our dataset
    
    # Create dataset and dataloader
    dataset = LatexImageDataset(
        mapping_file=args.mapping_file,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, vae, text_encoder = accelerator.prepare(
        unet, optimizer, train_dataloader, vae, text_encoder
    )
    
    # Calculate number of training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Setup LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Setup TensorBoard tracking if requested
    if args.with_tracking:
        accelerator.init_trackers("stable_diffusion_latex", config=vars(args))
    
    # Training loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        print(f"Number of examples: {len(dataset)}")
        print(f"Number of epochs: {args.num_train_epochs}")
        print(f"Instantaneous batch size per device: {args.train_batch_size}")
        print(f"Total train batch size (with parallel, distributed & accumulation): {total_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Total optimization steps: {max_train_steps}")
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training steps")
    global_step = 0
    
    # Initialize memory tracking
    if args.track_memory:
        torch.cuda.reset_peak_memory_stats()
    
    try:
        for epoch in range(args.num_train_epochs):
            unet.train()
            epoch_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space (this would be handled by VAE in the full pipeline)
                    # Here we'll work directly with pixel values for simplicity
                    pixel_values = batch["pixel_values"]
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    with torch.no_grad():
                        text_embeddings = text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )[0]
                    
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor  # Scale latents
                    # Sample noise that we'll add to the images
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
                    ).long()
                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Predict the noise residual using the properly formatted text embeddings
                    noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Gather the losses across all processes for logging (if we use distributed training)
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    epoch_loss += avg_loss.item() / args.gradient_accumulation_steps
                    
                    # Backpropagate
                    accelerator.backward(loss)
                    
                    # Clip gradients
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                        
                    # Update weights
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    progress_bar.set_postfix({"loss": epoch_loss})
                    
                    # Log metrics
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "train_loss": epoch_loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "step": global_step,
                            },
                            step=global_step,
                        )
                    
                    # Memory tracking
                    if args.track_memory and accelerator.is_main_process and step % 100 == 0:
                        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        print(f"Memory usage: {max_memory:.2f} GB")
                    
                    # Save model checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            print(f"Saved state to {save_path}")
                
                # Break if we've reached the maximum number of steps
                if global_step >= max_train_steps:
                    break
            
            # Log epoch metrics
            if accelerator.is_main_process:
                print(f"Epoch {epoch} completed. Average loss: {epoch_loss}")
        
        # Save the final model
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)
            print(f"Model saved to {args.output_dir}")
            
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Training error: {e}")
        raise e
    finally:
        accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stable Diffusion for LaTeX to image generation")
    
    # Data arguments
    parser.add_argument("--mapping_file", type=str, default="latex_equations.txt", help="Path to the mapping file")
    parser.add_argument("--image_dir", type=str, default="latex_images", help="Directory containing the LaTeX images")
    parser.add_argument("--output_dir", type=str, default="sd-latex-model", help="Output directory for the model")
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model identifier")
    parser.add_argument("--max_length", type=int, default=77, help="Max length of the text prompt")
    
    # Training arguments
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type for learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    
    # Misc
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], 
                        help="Mixed precision mode (recommended: fp16 for RTX series)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of updates steps to accumulate for a backward pass")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--with_tracking", action="store_true", help="Enable TensorBoard tracking")
    parser.add_argument("--track_memory", action="store_true", help="Track GPU memory usage")
    
    args = parser.parse_args()
    train(args)