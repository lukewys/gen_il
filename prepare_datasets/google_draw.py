import numpy as np
from tqdm import tqdm
import glob
import os

if __name__ == '__main__':
    data_dir = '/data/creativity_generation/generative_IL/data/google_draw/numpy_bitmap'
    output_dir = '/data/creativity_generation/generative_IL/data/google_draw'
    data_list = glob.glob(f'{data_dir}/*.npy')

    train_data = []
    test_data = []
    for data_path in tqdm(data_list):
        data = np.load(data_path)
        # split 1% of the data to test
        # split 10% of the data to train
        test_data.append(data[:int(data.shape[0] * 0.01)])
        train_data.append(data[int(data.shape[0] * 0.01):int(data.shape[0] * 0.11)])

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    print(train_data.shape)
    print(test_data.shape)

    np.save(os.path.join(output_dir, 'google_draw_train.npy'), train_data)
    np.save(os.path.join(output_dir, 'google_draw_test.npy'), test_data)
