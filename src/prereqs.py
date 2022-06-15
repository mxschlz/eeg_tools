import glob
import os
import pathlib
import re
import json
import mne
from matplotlib import pyplot as plt
_file_format= ".json"
_dir = pathlib.Path(os.getcwd())

# TODO: write function to get header files sorted by subject.
# TODO: write function to create config file.
# TODO: write function to create folders (figures,epochs,evokeds) according to specific structure.


def get_file(type="mapping", dir=_dir, format=_file_format):
    if type == "header":
        header_files = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".vhdr"):
                    header_files.append(os.path.join(root, file))
        if len(header_files) == 0:
            print("No .vhdr files found!")
        return header_files
    elif type == "montage":
        file_format = ".bvef"
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(f"{file_format}"):
                    file_path = str((os.path.join(root, file)))
                    montage = mne.channels.read_custom_montage(fname=file_path)
                    return montage
    elif type == "ica":
        file_format = ".fif"
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(f"{type}{file_type}"):
                    file_path = str((os.path.join(root, file)))
                    ica_ref = mne.preprocessing.read_ica(file_path)
                    return ica_ref
    else:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(f"{type}{format}"):
                    file_path = str((os.path.join(root, file)))
            with open(file_path) as file:
                lf = json.load(file)
            return lf

def get_ids(header_files):
    return ids


def make_raw(header_files, preload=True):
    raw_files = []
    for header_file in header_files:
        raw_files.append(mne.io.read_raw_brainvision(
            header_file, preload=preload))  # read BrainVision files.
    raw = mne.concatenate_raws(raw_files)  # make raw files
    return raw

def make_folders(dir):
    print("Folders successfully generated!")

def qc(step="rereference"):
    print("Sucessfully created preprocessing quality checks!")

def make_config():
    pass

if __name__ == "__main__":
    dir = pathlib.Path(os.getcwd())
    cfg = get_file("config")
    mapping = get_file("mapping")
    montage = get_file("montage")
    header_files = get_file("vhdr")
    ica_ref = get_file(type="ica", file_type=".fif")
