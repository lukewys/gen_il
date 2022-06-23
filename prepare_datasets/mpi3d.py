import numpy as np
import os

data_path = '/data/creativity_generation/generative_IL/data/mpi3d_realistic.npz'

data = np.load(data_path)['images']

random_index = np.random.choice(data.shape[0], size=130000, replace=False)
train_data = data[random_index][:100000]
test_data = data[random_index][100000:]

np.save(os.path.dirname(data_path) + '/mpi3d_realistic_subset_train.npy', train_data)
np.save(os.path.dirname(data_path) + '/mpi3d_realistic_subset_test.npy', test_data)
