import sys
sys.path.append("D:/Projects/eeg_tools/src/eeg_tools")
import pathlib
import setup_eeg_tools as set

# default prerequisites:
root_dir = pathlib.Path("D:/EEG/vocal_effort")
cfg = set.load_file("config", dir=root_dir)
ica_ref = set.load_file(type="ica", dir=root_dir)
mapping = set.load_file("mapping", dir=root_dir)
montage = set.load_file("montage", dir=root_dir)
