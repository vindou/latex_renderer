import os
import subprocess
import pandas as pd
import time

output_dir = "latex_images"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("latex_equations.csv")
mapping_file = "latex_equations.txt"

with open(mapping_file, "w") as f_map:
    for i, equation in enumerate(df["latex_equation"]):
        output_path = os.path.join(output_dir, f"equation_{i}.png")
        f_map.write(f"{i}||{equation}\n")
        base_name = f"equation_{i}"
        tex_filename = f"{base_name}.tex"
        pdf_filename = f"{base_name}.pdf"
        png_filename = output_path
        latex_doc = f"""
        \\documentclass{{standalone}}
        \\usepackage{{amsmath}}
        \\begin{{document}}
        ${equation}$
        \\end{{document}}
        """

        with open(tex_filename, "w") as f:
            f.write(latex_doc)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename], check=True)
        wait_time = 0
        while not os.path.exists(pdf_filename) and wait_time < 2:
            time.sleep(0.2)
            wait_time += 0.2

        if not os.path.exists(pdf_filename):
            print(f"❌ Error: PDF file {pdf_filename} not found!")
            continue

        subprocess.run(["convert", "-density", "300", pdf_filename, "-quality", "100", png_filename], check=True)
        os.rename(png_filename, os.path.join(output_dir, os.path.basename(png_filename)))

        for ext in [".aux", ".log", ".pdf", ".tex"]:
            try:
                os.remove(base_name + ext)
            except FileNotFoundError:
                pass

        print(f"✅ Rendered: {png_filename}")

print("🎉 Dataset generation complete! LaTeX equations saved to latex_equations.txt")
