import os
import glob

def process_file(filepath):
    if filepath.endswith("theme.py"):
        return
    with open(filepath, "r") as f:
        content = f.read()
    
    new_content = content.replace("}pt;", "}px;")
    new_content = new_content.replace("}pt", "}px")
    
    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Updated pt to px in {filepath}")

if __name__ == "__main__":
    for filepath in glob.glob("app/ui/**/*.py", recursive=True):
        process_file(filepath)
    print("Done refactoring pt to px.")
