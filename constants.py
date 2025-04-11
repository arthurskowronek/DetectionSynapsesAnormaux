import datetime
import numpy as np
from pathlib import Path
from numpy.random import RandomState, MT19937, SeedSequence


# DIRECTORYS
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_PKL_NAME = f'dataset_{DATE}.pkl'
DATASET_PKL_DIR = Path('./dataset_pkl')
DATA_DIR = Path('./data')
MUTANT_DIR = DATA_DIR / '_Mutant'
WT_DIR = DATA_DIR / '_WT'

# Constants
N_FEAT = 12
N_BINS_FEAT = 20
NUMBER_OF_PIXELS = 1024
IMAGE_SIZE = (NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)
MIN_AREA_COMPO = 0
MIN_AREA_FEATURE = 10

# TRAINING PARAMETERS
N_RUNS = 100
MAX_BINS = 255
LEARN_RATE = 0.1
MAX_ITER = 1000
IN_PARAM = np.array([MAX_BINS, LEARN_RATE, MAX_ITER], dtype='float')
SEED = RandomState(MT19937(SeedSequence(753))) # Set random seed for reproducibility

# FEATURE EXTRACTION
MIN_AREA_FEATURE = 10
MAX_LENGTH_OF_FEATURES = 20000