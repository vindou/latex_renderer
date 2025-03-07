import torch
import argparse
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

def generate_image_from_latex(latex_equation, model_path, output_path=None, device="cuda"):
    """
    Generate an image from a LaTeX equation using the fine-tuned Stable Diffusion model.
    
    Args:
        latex_equation (str): The LaTeX equation to render
        model_path (str): Path to the fine-tuned model
        output_path (str, optional): Path to save the generated image
        device (str): Device to run inference on ("cuda" or "cpu")
    
    Returns:
        PIL.Image: The generated image
    """
    # Load the pipeline with the fine-tuned model
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe = pipe.to(device)
    
    # Generate the image
    with torch.no_grad():
        image = pipe(latex_equation).images[0]
    
    # Save the image if an output path is provided
    if output_path:
        image.save(output_path)
        print(f"Image saved to {output_path}")
    
    return image

def generate_images_from_file(input_file, model_path, output_dir, device="cuda"):
    """
    Generate images from a file containing LaTeX equations.
    
    Args:
        input_file (str): Path to the file containing LaTeX equations (one per line)
        model_path (str): Path to the fine-tuned model
        output_dir (str): Directory to save the generated images
        device (str): Device to run inference on ("cuda" or "cpu")
    """
    import os
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pipeline with the fine-tuned model
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe = pipe.to(device)
    
    # Read equations from the file
    with open(input_file, 'r') as f:
        equations = [line.strip().split('||')[1] if '||' in line else line.strip() for line in f]
    
    # Generate images for each equation
    for i, equation in enumerate(tqdm(equations, desc="Generating images")):
        output_path = os.path.join(output_dir, f"generated_{i}.png")
        with torch.no_grad():
            image = pipe(equation).images[0]
            image.save(output_path)
    
    print(f"Generated {len(equations)} images in {output_dir}")

def compare_generated_with_original(generated_dir, original_dir, mapping_file, num_samples=5):
    """
    Compare generated images with the original images.
    
    Args:
        generated_dir (str): Directory containing generated images
        original_dir (str): Directory containing original images
        mapping_file (str): Path to the mapping file
        num_samples (int): Number of samples to display
    """
    import os
    import random
    import matplotlib.pyplot as plt
    
    # Load the mapping
    equations = []
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split('||')
            if len(parts) >= 2:
                idx, equation = parts
                equations.append((idx, equation))
    
    # Select random samples
    samples = random.sample(equations, min(num_samples, len(equations)))
    
    # Display the comparisons
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    
    for i, (idx, equation) in enumerate(samples):
        # Original image
        original_path = os.path.join(original_dir, f"equation_{idx}.png")
        original_img = Image.open(original_path).convert("RGB")
        
        # Generated image
        generated_path = os.path.join(generated_dir, f"generated_{i}.png")
        generated_img = Image.open(generated_path).convert("RGB")
        
        # Display images
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(generated_img)
        axes[i, 1].set_title("Generated")
        axes[i, 1].axis("off")
        
        # Display equation
        plt.figtext(0.5, 0.01 + i * (1/num_samples), f"Equation: {equation}", ha="center")
    
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from LaTeX equations using Stable Diffusion")
    
    parser.add_argument("--mode", type=str, choices=["single", "batch", "compare"], default="single",
                        help="Mode of operation: single equation, batch processing, or comparison")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--latex", type=str, help="LaTeX equation (for single mode)")
    parser.add_argument("--input_file", type=str, help="Input file containing LaTeX equations (for batch mode)")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory for generated images")
    parser.add_argument("--original_dir", type=str, help="Directory containing original images (for comparison)")
    parser.add_argument("--mapping_file", type=str, help="Path to the mapping file (for comparison)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to compare (for comparison)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on (cuda or cpu)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.latex:
            parser.error("--latex is required for single mode")
        image = generate_image_from_latex(args.latex, args.model_path, 
                                         output_path=f"{args.output_dir}/single_output.png",
                                         device=args.device)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Generated Image for: {args.latex}")
        plt.show()
    
    elif args.mode == "batch":
        if not args.input_file:
            parser.error("--input_file is required for batch mode")
        generate_images_from_file(args.input_file, args.model_path, args.output_dir, device=args.device)
    
    elif args.mode == "compare":
        if not args.original_dir or not args.mapping_file:
            parser.error("--original_dir and --mapping_file are required for compare mode")
        compare_generated_with_original(args.output_dir, args.original_dir, args.mapping_file, args.num_samples)