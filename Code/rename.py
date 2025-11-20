import os

'''
This script was used to create the dataset. It renames files.
'''

folder = "cells"  # folder containing your images

for filename in os.listdir(folder):
    # Only process files (skip subfolders)
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path):
        name, ext = os.path.splitext(filename)
        new_name = f"{name}g{ext}"
        new_path = os.path.join(folder, new_name)
        os.rename(file_path, new_path)

print("Renaming complete!")
