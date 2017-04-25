import pandas as pd
import numpy as np

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

print(train)
