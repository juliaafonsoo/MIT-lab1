# part1
# Copyright 2024 MIT Introduction to Deep Learning. All Rights Reserved.
#
# Licensed under the MIT License. You may not use this file except in compliance
# with the License. Use and/or modification of this code outside of MIT Introduction
# to Deep Learning must reference:
#
# © MIT Introduction to Deep Learning
# http://introtodeeplearning.com
#

#setup 
import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
import tensorflow as tf
from tqdm import tqdm
from scipy.io.wavfile import write

import comet_ml
COMET_API_KEY = "NAYItnCcXRFTkkSkCfrGwH8jb"

assert len(tf.config.list_physical_devices('GPU')) > 0

# Download the dataset
songs = mdl.lab1.load_training_data()

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)

