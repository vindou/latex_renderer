import os
import json
import re
from PIL import Image

# Define dataset paths
image_dir = "latex_images"  # Folder where images are stored
mapping_file = "latex_equations.txt"  # LaTeX equation mappings
dataset_json = "latex_dataset.json"  # Output JSON for training

# Ensure processed image directory exists
processed_image_dir = os.path.join(image_dir, "processed")
os.makedirs(processed_image_dir, exist_ok=True)

# Read LaTeX equations from mapping file
latex_equations = {}
with open(mapping_file, "r") as f_map:
    for line in f_map:
        index, equation = line.strip().split("||", 1)
        latex_equations[int(index)] = equation  # Store in dictionary

# Regex pattern to extract index from filenames (e.g., equation_123.png -> 123)
filename_pattern = re.compile(r"equation_(\d+)\.png")

# Initialize dataset list
dataset_entries = []

# Target size for Stable Diffusion training
target_size = (512, 512)

# Scan for PNG images in latex_images/
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# Process images
for image_file in image_files:
    match = filename_pattern.match(image_file)
    if match:
        index = int(match.group(1))  # Get the equation index
        latex_equation = latex_equations.get(index, "UNKNOWN")  # Retrieve equation

        # Load image
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        
        # Resize to 512x512 for Stable Diffusion training
        img = img.resize(target_size, Image.LANCZOS)

        # Save processed image
        processed_path = os.path.join(processed_image_dir, image_file)
        img.save(processed_path, format="PNG")

        # Add to dataset
        dataset_entries.append({"latex_equation": latex_equation, "image_path": processed_path})

# Save dataset in JSON format for Stable Diffusion training
with open(dataset_json, "w") as f:
    json.dump(dataset_entries, f, indent=4)

print(f"✅ Dataset preprocessing complete! JSON saved as {dataset_json}")
print(f"Processed images stored in: {processed_image_dir}")
