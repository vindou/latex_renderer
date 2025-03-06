import random
import pandas as pd

# Define LaTeX equation components
operators = ['+', '-', '*', '/', '^']
functions = ['\\sin', '\\cos', '\\tan', '\\exp', '\\log']
greek_letters = ['\\alpha', '\\beta', '\\gamma', '\\delta', '\\lambda', '\\theta']
variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
constants = ['e', 'pi']

# Function to generate a simple algebraic equation
def generate_simple_equation():
    var1 = random.choice(variables)
    var2 = random.choice(variables)
    op = random.choice(operators)
    return f"{var1} {op} {var2}"

# Function to generate an equation with a function
def generate_function_equation():
    func = random.choice(functions)
    var = random.choice(variables)
    return f"{func}({var})"

# Function to generate an equation with an integral
def generate_integral_equation():
    var = random.choice(variables)
    return f"\\int {var} d{var}"

# Function to generate an equation with a summation
def generate_summation_equation():
    var = random.choice(variables)
    return f"\\sum_{{i=1}}^n {var}_i"

# Function to generate a matrix equation
def generate_matrix_equation():
    return "\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}"

# Function to generate random equations
def generate_random_equation():
    choice = random.randint(1, 5)
    if choice == 1:
        return generate_simple_equation()
    elif choice == 2:
        return generate_function_equation()
    elif choice == 3:
        return generate_integral_equation()
    elif choice == 4:
        return generate_summation_equation()
    elif choice == 5:
        return generate_matrix_equation()

# Function to generate 'n' LaTeX equations
def generate_latex_equations(n):
    return [generate_random_equation() for _ in range(n)]

# Example Usage
if __name__ == "__main__":
    equations = generate_latex_equations(100)
    
    # Save to CSV
    df = pd.DataFrame(equations, columns=['latex_equation'])
    df.to_csv("latex_equations.csv", index=False)
    
    # Print sample output
    for eq in equations[:10]:  # Print first 10 equations
        print(eq)
