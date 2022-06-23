import os
from tqdm import tqdm
import argparse
import pandas
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/data/creativity_generation/generative_IL/data/wikiart_dataset/wikiart')
    parser.add_argument('--metadata_dir', type=str,
                        default='/data/creativity_generation/generative_IL/data/wikiart_dataset')
    parser.add_argument('--category', type=str, default='genre')
    parser.add_argument('--output_dir', type=str,
                        default='/data/creativity_generation/generative_IL/data/wikiart_dataset/wikiart_split')

    args = parser.parse_args()

    train_metadata = pandas.read_csv(os.path.join(args.metadata_dir, f'{args.category}_train.csv'))
    test_metadata = pandas.read_csv(os.path.join(args.metadata_dir, f'{args.category}_val.csv'))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    for i, row in tqdm(train_metadata.iterrows(), total=train_metadata.shape[0]):
        file_path = row[0]
        dir_name = os.path.dirname(file_path)
        os.makedirs(os.path.join(args.output_dir, 'train', dir_name), exist_ok=True)
        shutil.copy(os.path.join(args.data_dir, row[0]),
                    os.path.join(args.output_dir, 'train', dir_name, os.path.basename(file_path)))

    for i, row in tqdm(test_metadata.iterrows(), total=test_metadata.shape[0]):
        file_path = row[0]
        dir_name = os.path.dirname(file_path)
        os.makedirs(os.path.join(args.output_dir, 'test', dir_name), exist_ok=True)
        shutil.copy(os.path.join(args.data_dir, row[0]),
                    os.path.join(args.output_dir, 'test', dir_name, os.path.basename(file_path)))