#!/usr/bin/env python3

import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_classes.py <train_dir>")
        sys.exit(1)

    train_dir = sys.argv[1]

    # List all immediate subfolders (each subfolder is a class)
    class_names = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    
    # Sort them alphabetically
    class_names = sorted(class_names)

    # Print them; you can copy this output directly into your script as a Python list
    print("class_names = [")
    for name in class_names:
        print(f'    "{name}",')
    print("]")

if __name__ == "__main__":
    main()
