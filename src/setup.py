import glob
import os
import pathlib
import re
import json

# TODO: write function to get header files sorted by subject.
# TODO: write function to create config file.
# TODO: write function to create folders (figures,epochs,evokeds) according to specific structure.


DIR = pathlib.Path(os.getcwd())


def get_header_files(DIR):
    header_files = []
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if file.endswith(".vhdr"):
                header_files.append(os.path.join(root, file))
    return header_files

def load_file(type="mapping")
    if type == "config":
        for root, dirs, files in os.walk(DIR):
            for file in files:
                if file.endswith("config.json") or file.endswith("cfg.json"):
                    config_path = str((os.path.join(root, file)))
        with open(config_path) as file:
            cfg = json.load(file)
        return cfg
    if type == "mapping":
        for root, dirs, files in os.walk(DIR):
            for file in files:
                if file.endswith("mapping.json"):
                    mapping_path = str((os.path.join(root, file)))
        with open(mapping_path) as file:
            mapping = json.load(file)
        return mapping
    if type == "folder_structure":
        for root, dirs, files in os.walk(DIR):
            for file in files:
                if file.endswith("config.json") or file.endswith("cfg.json"):
                    config_path = str((os.path.join(root, file)))
        with open(config_path) as file:
            cfg = json.load(file)


def make_folder_structure():

if __name__ == "__main__":
    header_files = get_header_files(DIR)
    cfg = load_config()
    mapping = load_mapping()
