import h5py
import os
from pathlib import Path

name_to_find = "BA_91176"
data_path = Path("/home/teusink/code/DiffSBDD/data/pmhc/raw_files")


for file in os.listdir(data_path):
    content = h5py.File(data_path / file, "r")
    for i, (name, _) in enumerate(content.items()):
        if name == name_to_find:
            print(file)
            print(i)
            break