import os
import sys

output_dir = "test_output"

print(f"Attempting to create directory: {output_dir}")
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory {output_dir} should now exist.")
    if os.path.isdir(output_dir):
        print(f"Successfully verified {output_dir} exists.")
    else:
        print(f"ERROR: os.makedirs ran but {output_dir} does not exist.")
        sys.exit(1) # Exit with error
except Exception as e:
    print(f"ERROR: Failed to create directory {output_dir}: {e}")
    sys.exit(1) # Exit with error

# Create a dummy file in the directory
try:
    with open(os.path.join(output_dir, "test_file.txt"), "w") as f:
        f.write("test")
    print(f"Successfully created test_file.txt in {output_dir}")
except Exception as e:
    print(f"ERROR: Failed to create test_file.txt in {output_dir}: {e}")
    sys.exit(1) # Exit with error

print("Simple script finished successfully.")
sys.exit(0)
