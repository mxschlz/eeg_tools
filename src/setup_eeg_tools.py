import glob
import os
import pathlib
import re
import json
import mne
from matplotlib import pyplot as plt
_file_format = ".json"
_dir = pathlib.Path(os.getcwd())


# TODO: fix search().
# TODO: implement search function into other functions such as load_file().
# TODO: make get_ids() more elegant so that reg ex patterns are found more easily.
# TODO: fix read_object().

def search(dir, format):
    files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(format):
                files.append(os.path.join(root, file))
    return files


def load_file(type="mapping", dir=_dir, format=_file_format):
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
                if file.endswith(f"{type}{file_format}"):
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


def save_file(file):
    pass

# r"" == raw string
# \b matches on a change from a \w (a word character) to a \W (non word character)
# \w{6} == six alphanumerical characters
# RegEx expression to match subject ids (6 alphanumerical characters)
def get_ids(header_files, pattern=r'\b\w{6}\b'):
    ids = []
    for header_file in header_files:
        match = re.search(pattern=pattern, string=header_file)
        if match.group() not in ids:
            ids.append(match.group())
    return ids


def make_folders(root_dir, id=None, folders=["epochs", "raw", "evokeds", "figures"]):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            for folder in folders:
                folder_path = pathlib.Path(root) / folder
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)
                    print(f"{folder} folder in {id} successfully generated!")
                else:
                    print(f"{folder} folder already exists in {id}!")
        else:
            continue


def save_object(data, root_dir, id, overwrite=True):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            if isinstance(data, mne.io.brainvision.brainvision.RawBrainVision):
                folder_path = pathlib.Path(root) / "raw"
                data.save(f"{folder_path}\\{id}_raw.fif", overwrite=overwrite)
                break
            if isinstance(data, mne.Epochs):
                folder_path = pathlib.Path(root) / "epochs"
                data.save(f"{folder_path}\\{id}-epo.fif", overwrite=overwrite)
                break
            if isinstance(data, mne.Evoked) or isinstance(data, list):
                folder_path = pathlib.Path(root) / "evokeds"
                if isinstance(data, list):
                    mne.write_evokeds(
                        f"{folder_path}\\{id}-ave.fif", data, overwrite=overwrite)
                else:
                    data.save(f"{folder_path}\\{id}-ave.fif",
                              overwrite=overwrite)
                break
            else:
                print("Data needs to be an mne object of type mne.io.Raw, mne.Epochs or mne.Evoked!")


def read_object(data_type, root_dir, id):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            if data_type == "raw":
                folder_path = pathlib.Path(root) / "raw"
                raw = mne.io.read_raw_fif(f"{folder_path}\\{id}_raw.fif", preload=True)
                return raw
            if data_type == "epochs":
                folder_path = pathlib.Path(root) / "epochs"
                epochs = mne.read_epochs(f"{folder_path}\\{id}-epo.fif", preload=True)
                return epochs
            if data_type == "evokeds":
                folder_path = pathlib.Path(root) / "evokeds"
                evokeds = mne.read_epochs(f"{folder_path}\\{id}-ave.fif", preload=True)
                return evokeds

def check_id(id, root_dir):
    for root, dirs, files in os.walk(root_dir):
        if root.endswith(id):
            evokeds_fname = pathlib.Path(root) / "evokeds" / f"{id}-ave.fif"
            if not os.path.exists(evokeds_fname):
                print(f"Subject has not been processed yet.")
                # return False
            else:
                print(f"{id} has been processed already.")
                # return True

def make_config(config_dir, config_format=".json"):
    pass


if __name__ == "__main__":  # workflow
    root_dir = pathlib.Path("D:/EEG")
    cfg = load_file("config")
    mapping = load_file("mapping")
    montage = load_file("montage")
    header_files = load_file("header", dir=root_dir)
    ica_ref = load_file(type="ica", format=".fif")
    ids = get_ids(header_files=header_files)
    id = ids[0]
    for id in ids[:1]:
        make_folders(root_dir=root_dir, id=id)
        raw = make_raw(header_files, id, mapping=mapping, montage=montage)
        save_object(raw, root_dir, id)
        preprocessing.run_pipeline()
