from setuptools import setup

setup(name="eeg_tools",
      version="0.0.1",
      description="Package to plot, analyze and visualize electrophysiological (EEG) data.",
      py_modules=["mne"],
      package_dir={"": "src"})
