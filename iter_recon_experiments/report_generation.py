import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
from PIL import Image
import numpy as np

data_dir = '/Users/yusongwu/Desktop/实验记录/8.31'
output_dir = '/Users/yusongwu/Desktop/实验记录/8.31/results_summary'
model_list = glob.glob(data_dir + '/*/')

for model_dir in tqdm(model_list):
    os.makedirs(os.path.join(output_dir, os.path.basename(os.path.dirname(model_dir))), exist_ok=True)
    experiment_list = glob.glob(model_dir + '/*/')
    for experiment_dir in experiment_list:
        experiment_name = os.path.basename(os.path.dirname(experiment_dir))

        image_list = sorted(glob.glob(experiment_dir + '/*_small.png'), key=os.path.getmtime)
        if len(image_list) > 3:

            image_index = [0, 1, 2, 3, int(len(image_list) * 0.2) - 1, int(len(image_list) * 0.5) - 1,
                           int(len(image_list) * 0.8) - 1, len(image_list) - 1]

            num_row = 2
            num_col = 4
            fig, axes = plt.subplots(num_row, num_col, dpi=250)  # , figsize=(1.5*num_col,2*num_row))

            for i, idx in enumerate(image_index):
                file = image_list[idx]
                pil_im = Image.open(file, 'r')
                image = np.asarray(pil_im)
                image = image[:image.shape[0] // 2, :image.shape[1] // 2, :]
                ax = axes[i // num_col, i % num_col]
                ax.imshow(image)
                ax.axis('off')
                ax.set_title(f'Epoch {idx + 1}')
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'{os.path.join(output_dir, os.path.basename(os.path.dirname(model_dir)))}/{experiment_name}.png')
            plt.close()
