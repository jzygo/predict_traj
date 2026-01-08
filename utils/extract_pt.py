import torch
import sys
import os

def format_value(value, indent_level=0):
    """
    Recursively format the value to be human-readable.
    """
    indent = "    " * indent_level
    
    if isinstance(value, torch.Tensor):
        # Extract tensor metadata and data
        info = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
        return f"{info}\n{indent}Values: {value.tolist()}"
            
    elif isinstance(value, dict):
        lines = []
        for k, v in value.items():
            lines.append(f"\n{indent}{k}: {format_value(v, indent_level + 1)}")
        return "".join(lines)
        
    elif isinstance(value, (list, tuple)):
        lines = [f"Sequence (len={len(value)})"]
        # Limit display for very long lists
        limit = 10000 
        for i, v in enumerate(value):
            if i >= limit:
                lines.append(f"\n{indent}... ({len(value) - limit} more items)")
                break
            lines.append(f"\n{indent}[{i}]: {format_value(v, indent_level + 1)}")
        return "".join(lines)
        
    else:
        return str(value)

def extract_pt_to_txt(pt_path, txt_path):
    print(f"Loading {pt_path}...")
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Formatting and writing to file...")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Content of {os.path.basename(pt_path)} ===\n")
        f.write(format_value(data))
        f.write("\n==========================================\n")
    
    print(f"Successfully saved to {txt_path}")

if __name__ == "__main__":
    # Example Usage: Replace these paths with your actual file paths
    input_pt_file = "../src/simulation/stroke_visualizations/joint_trajectory.pt"
    output_txt_file = "output.txt"
    
    # You can also pass arguments via command line
    extract_pt_to_txt(input_pt_file, output_txt_file)