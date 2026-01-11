import inspect
import re

def parse_numpy_docstring(docstring):
    """
    Extracts parameter details from a NumPy-style docstring.
    Returns a dictionary with parameter names as keys and their details.
    """
    param_section = re.search(r"Parameters\n[-]+\n(.*?)(\n\nAttributes|\n\nMethods|\Z)", docstring, re.DOTALL)
    param_info = {}

    if param_section:
        param_text = param_section.group(1).strip()
        lines = param_text.split("\n")
        current_param = None
        current_description = []

        for line in lines:
            if re.match(r"^\s", line):
                # Line is part of the current parameter's description if starts with whitespace
                if current_param:
                    current_description.append(line.strip())
            else:
                # Line starts a new parameter
                if current_param:
                    param_info[current_param[0]] = {
                        "type": current_param[1].strip(),
                        "default": current_param[2].strip() if current_param[2] else None,
                        "description": " ".join(current_description)
                    }
                # Match "param_name : type, default=value"
                match = re.match(r"(\w+)\s*:\s*([^,]+)(?:,\s*default\s*=\s*(.*))?", line)
                if match:
                    current_param = match.groups()
                    current_description = []

        # Add the last parameter
        if current_param:
            param_info[current_param[0]] = {
                "type": current_param[1].strip(),
                "default": current_param[2].strip() if current_param[2] else None,
                "description": " ".join(current_description)
            }

    return param_info

def generate_markdown_table(func):
    """
    Extracts parameter details from a NumPy-style docstring and function signature,
    then generates a Markdown table.
    """
    # signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    # Parse docstring
    param_info = parse_numpy_docstring(docstring)

    # Create the table header
    markdown_table = "| Option  | Type  | Default Value | Description |\n"
    markdown_table += "|---------|------|--------------|-------------|\n"

    for param_name in param_info.keys():
        # if param_name not in signature.parameters:
        # param_type = param_info.get(param_name, {}).get("type", "Unknown")
        param_type = param_info[param_name]["type"]
        # default_value = param_info.get(param_name, {}).get("default", "No Default")
        default_value = param_info[param_name]["default"]
        # description = param_info.get(param_name, {}).get("description", "No description provided")
        description = param_info[param_name]["description"]
        markdown_table += f"| {param_name} | {param_type} | {default_value} | {description} |\n"

    return markdown_table

def generate_educational_algorithm_tables(config):
    from modopt import (SteepestDescent, Newton, QuasiNewton, 
                        NewtonLagrange, L2PenaltyEq, 
                        SQP, InteriorPoint,
                        NelderMeadSimplex, PSO,
                        SimulatedAnnealing)

    # Generate the markdown table for each educational algorithm
    for algorithm in [SteepestDescent, Newton, QuasiNewton, NewtonLagrange, L2PenaltyEq, 
                      InteriorPoint, SQP, NelderMeadSimplex, PSO, SimulatedAnnealing]:
        markdown_output = generate_markdown_table(algorithm)

        # write the markdown table to a file
        with open(config['target'] + f"{algorithm.__name__}_options_table.md", "w") as f:
            f.write(markdown_output)