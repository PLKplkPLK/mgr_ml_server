import os
import sys
import threading

import torch
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps

from pipeline import run_megadetector, run_pipeline


run_megadetector(folder, batch_md, workers)
run_pipeline(batch_df)

"""
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
"""