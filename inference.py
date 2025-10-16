import os
import torch
import argparse
from options.test_options import TestOptions
import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset_testing
from torch.utils.data import DataLoader
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime
from models import create_model
from test import inference

# ----------------------------
# Parse command-line arguments
# ----------------------------
opt = TestOptions().parse()

input_dir = opt.input
output_dir = opt.output
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Collect all input .nii.gz files from 64mT folders
# ----------------------------

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

test_files = []
for file in os.listdir(input_dir):
    if file.endswith(".nii.gz") or file.endswith(".nii"):
        test_files.append(os.path.join(input_dir, file))
test_files = sorted(test_files, key=numericalSort)

# ----------------------------
# Collect all subject IDs and make output folders for each subject
# ----------------------------

subject_IDs = []
subject_IDs_temp = os.listdir(input_dir)
for i in subject_IDs_temp:
    subject_ID = i.split('.')[0]
    subject_IDs.append(subject_ID)
subject_IDs = sorted(subject_IDs, key=numericalSort)

# ----------------------------
# Load model
# ----------------------------


model_T2 = create_model(opt)
model_T2.setup(opt)

# # ----------------------------
# # Run T2 inference and save outputs
# # ----------------------------


num_subs = int(len(test_files))

for i in range(num_subs):
    input_file = test_files[i]
    subject_ID = subject_IDs[i]

    output_name = os.path.join(output_dir, subject_ID + "_GAMBAS.nii.gz")

    inference(model_T2, input_file, output_name, False, (0.45, 0.45, 0.45), 128,
                128, 128, 32, 32, 1)

print("Inference completed. All outputs saved to:", output_dir)