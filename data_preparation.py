import os
import pandas as pd
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import json
from tqdm import tqdm

def prepare_dataset(mapping_file, image_dir, output_dir, max_samples=None):
    """
    Prepare a dataset for fine-tuning Stable Diffusion on LaTeX equations.

    Args:
        mapping_file (str): Path to the file mapping equation IDs to LaTeX equations
        image_dir (str): Directory containing the LaTeX equation images
        output_dir (str): Directory to save the processed dataset
        max_samples (int, optional): Maximum number of samples to include
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Load the mapping file
    equations = []
    with open(mapping_file, 'r') as f:
        for line in f:
            if '||' in line:
                idx, equation = line.strip().split('||')
                equations.append((idx, equation))

    # Limit samples if specified
    if max_samples is not None:
        equations = equations[:max_samples]

    # Initialize the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Process each sample
    metadata = []
    for idx, equation in tqdm(equations, desc="Processing samples"):
        image_path = os.path.join(image_dir, f"equation_{idx}.png")
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            continue

        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Save processed image
            output_image_path = os.path.join(output_dir, "images", f"equation_{idx}.png")
            image = image.resize((512, 512))
            image.save(output_image_path)

            # Add to metadata
            metadata.append({
                "file_name": f"images/equation_{idx}.png",
                "text": equation
            })
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Save metadata
    with open(os.path.join(output_dir, "metadata.jsonl"), 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

    print(f"Prepared dataset with {len(metadata)} samples")
    print(f"Dataset saved to {output_dir}")

def analyze_dataset(dataset_dir):
    """
    Analyze the prepared dataset.

    Args:
        dataset_dir (str): Directory containing the processed dataset
    """
    # Load metadata
    metadata_file = os.path.join(dataset_dir, "metadata.jsonl")
    texts = []

    with open(metadata_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])

    # Analyze text lengths
    text_lengths = [len(text) for text in texts]

    print(f"Dataset size: {len(texts)} samples")
    print(f"Average text length: {sum(text_lengths) / len(text_lengths):.2f} characters")
    print(f"Min text length: {min(text_lengths)} characters")
    print(f"Max text length: {max(text_lengths)} characters")

    # Analyze image statistics (optional)
    images_dir = os.path.join(dataset_dir, "images")
    image_files = os.listdir(images_dir)

    if len(image_files) > 0:
        sample_image = Image.open(os.path.join(images_dir, image_files[0]))
        print(f"Image dimensions: {sample_image.size}")

    # Count equation types
    type_count = {
        "matrix": sum(1 for text in texts if "\\begin{bmatrix}" in text),
        "integral": sum(1 for text in texts if "\\int" in text),
        "summation": sum(1 for text in texts if "\\sum" in text),
        "function": sum(1 for text in texts if any(f in text for f in ["\\sin", "\\cos", "\\tan", "\\exp", "\\log"]))
    }

    print("\nEquation type distribution:")
    for eq_type, count in type_count.items():
        print(f"  {eq_type}: {count} ({count/len(texts)*100:.2f}%)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for Stable Diffusion fine-tuning")
    parser.add_argument("--mapping_file", type=str, default="latex_equations.txt", help="Path to the mapping file")
    parser.add_argument("--image_dir", type=str, default="latex_images", help="Directory containing LaTeX images")
    parser.add_argument("--output_dir", type=str, default="processed_dataset", help="Output directory for processed dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--analyze", action="store_true", help="Analyze the dataset after preparation")

    args = parser.parse_args()

    prepare_dataset(args.mapping_file, args.image_dir, args.output_dir, args.max_samples)

    if args.analyze:
        analyze_dataset(args.output_dir)