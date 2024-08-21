import numpy as np
import os
import os.path as op
import glob
import shutil
import pandas as pd
import re

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


dolphin_folder = '/media/hdd/levibaljer/Dolphin_new/Workstation'
dolphin_subs = os.listdir(dolphin_folder)
for i in dolphin_subs:
    if 'DS' in i:
        dolphin_subs.remove(i)
dolphin_subs = sorted(dolphin_subs, key=numericalSort)

print(dolphin_subs)

current_num = 41
new_folder = '/media/hdd/levibaljer/spm_Dolphin/test'
for i in dolphin_subs[40:]:
    sub_folder = op.join(dolphin_folder, i)
    label = op.join(sub_folder, 'GT_ss.nii')
    image = op.join(sub_folder, 'AXI_ss.nii')

    label_folder = op.join(new_folder, 'labels')
    image_folder = op.join(new_folder, 'images')

    shutil.copy2(label, op.join(label_folder, str(current_num) + '.nii'))
    shutil.copy2(image, op.join(image_folder, str(current_num) + '.nii'))

    current_num += 1

# a = '/media/hdd/levibaljer/spm_Dolphin'

