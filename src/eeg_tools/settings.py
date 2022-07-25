import sys
sys.path.append("D:/Projects/eeg_tools/src/eeg_tools")
import pathlib
import setup_eeg_tools as set

# TODO: change root_dir so that it is universally suited.

# default prerequisites:
root_dir = pathlib.Path("D:/EEG/vocal_effort")
cfg = set.load_file(root_dir, "config")
ica_ref = set.load_file(root_dir, type="ica")
mapping = set.load_file(root_dir, "mapping")
montage = set.load_file(root_dir, "montage")
header_files = set.find(path=root_dir, mode="pattern", pattern="*.vhdr")
ids = set.get_ids(header_files=header_files, pattern=r'\b\w{6}\b')
