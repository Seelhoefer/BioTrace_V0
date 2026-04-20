import os
import glob

replacements = {
    "FONT_SIZE_SMALL": "FONT_SMALL",
    "FONT_SIZE_BODY": "FONT_BODY",
    "FONT_SIZE_SUBHEADING": "FONT_SUBTITLE",
    "FONT_SIZE_HEADING": "FONT_HEADING_2",
    "FONT_SIZE_TITLE": "FONT_TITLE",
    "FONT_SIZE_METRIC": "FONT_METRIC_XL",
    "SPACING_SMALL": "SPACE_1",
    "SPACING_MEDIUM": "SPACE_2",
    "SPACING_LARGE": "SPACE_3",
    "COLOR_PRIMARY_DARK": "COLOR_PRIMARY_HOVER",
}

def process_file(filepath):
    if filepath.endswith("theme.py"):
        return
    with open(filepath, "r") as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Updated {filepath}")

if __name__ == "__main__":
    for filepath in glob.glob("app/ui/**/*.py", recursive=True):
        process_file(filepath)
    print("Done refactoring.")
