import os
import torch
import numpy as np
from data_utils.process import choose_device

def load_dir(path, start, end):
    device = choose_device()  # Use the choose_device function to determine the appropriate device
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + ".lms")):
            lms = np.loadtxt(os.path.join(path, str(i) + ".lms"), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(path, str(i) + ".jpg"))
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).to(device)  # Move the tensor to the chosen device
    return lmss, imgs_paths