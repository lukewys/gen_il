import os
import argparse
import numpy as np
import torch

from train_utils import str2bool, set_seed

set_seed(1234)

import vae_utils
import gan_utils
import wta_utils


def train_model(model_assets, train_data, train_fn, iteration):
    print(f'=======Iteration {iteration}: Train Model=======')
    model_assets = train_fn(model_assets, train_data, train_extend)
    return model_assets


def train_model_with_teacher(new_model_assets, old_model_assets, train_with_teacher_fn, iteration):
    print(f'=======Iteration {iteration}: Train Model With Teacher=======')
    total_steps = 3000
    total_steps = int(np.round(train_extend * total_steps))
    model_assets = train_with_teacher_fn(new_model_assets, old_model_assets, total_steps)
    return model_assets


def generate_data(model_assets, gen_fn, iteration):
    print(f'=======Iteration {iteration}: Generate Data=======')
    data_generated = gen_fn(model_assets, gen_batch_size, gen_num_batch)
    # data_generated: tensor of [batch_size, *]
    return data_generated


def save_generated_data(model_assets, iteration):
    # data_generated: tensor of [batch_size, *]
    save_sample_fn(model_assets, log_dir, iteration)


def generative_iterated_learning(model_assets, train_data, train_fn, gen_fn, total_iterations):
    for iteration in range(1, total_iterations + 1):
        if train_with_teacher and iteration > 1:
            model_assets = train_model_with_teacher(new_model_assets, model_assets, train_with_teacher_fn, iteration)
        else:
            model_assets = train_model(model_assets, train_data, train_fn, iteration)
            data_generated = generate_data(model_assets, gen_fn, iteration)
            train_data = get_train_data_next_iter(train_data, data_generated, add_old_dataset=add_old_dataset)
            if args.dataset_keep_portion < 1.0:
                train_data = drop_dataset(train_data, args.dataset_keep_portion)
        print(f'=======Iteration {iteration}: Get New Model=======')
        new_model_assets = get_model_assets_next_iter(model_assets=model_assets)
        if iteration % save_image_interval == 0:
            save_generated_data(model_assets, iteration)


def drop_dataset(dataset, keep_portion):
    data = dataset.data.numpy()
    targets = dataset.targets.numpy()

    data = data[np.random.choice(data.shape[0], int(data.shape[0] * keep_portion), replace=False), ...]
    targets = targets[np.random.choice(targets.shape[0], int(targets.shape[0] * keep_portion), replace=False), ...]

    dataset.data = torch.tensor(data)
    dataset.targets = torch.tensor(targets)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gen IL')
    parser.add_argument('--model_type', type=str, default='vae', metavar='N',
                        help='model_type')
    parser.add_argument('--save_image_interval', type=int, default=1, metavar='N',
                        help='save_image_interval')
    parser.add_argument('--total_iterations', type=int, default=10, metavar='N',
                        help='total_iterations')
    parser.add_argument('--add_old_dataset', type=str2bool, nargs='?', const=True,
                        default=False, help='add old dataset')
    parser.add_argument('--train_extend', type=float, default=1.0, metavar='S',
                        help='train_extend')
    parser.add_argument('--gen_num_batch', type=int, default=12, metavar='N',
                        help='gen_num_batch')
    parser.add_argument('--train_with_teacher', type=str2bool, nargs='?', const=True,
                        default=False, help='train_with_teacher')
    parser.add_argument('--gan_filter_portion_max', type=float, default=None, metavar='S',
                        help='gan_filter_portion_max')
    parser.add_argument('--gan_filter_portion_min', type=float, default=None, metavar='S',
                        help='gan_filter_portion_min')
    parser.add_argument('--dataset_keep_portion', type=float, default=1.0, metavar='S',
                        help='dataset_keep_portion')
    args = parser.parse_args()

    save_image_interval = args.save_image_interval
    total_iterations = args.total_iterations
    train_extend = args.train_extend  # % of training steps
    add_old_dataset = args.add_old_dataset
    gen_batch_size = 500
    gen_num_batch = args.gen_num_batch  # for matching the size of mnist train data, =12
    model_type = args.model_type  # gan, vae, wta
    train_with_teacher = args.train_with_teacher

    log_dir = f'gen_il_logs/{model_type}_total_iter_{total_iterations}_train_extend_{train_extend}_' \
              f'gen_num_batch_{gen_num_batch}_add_old_dataset_{add_old_dataset}_train_with_teacher_{train_with_teacher}'
    if args.gan_filter_portion_max and args.gan_filter_portion_min:
        log_dir += f'_gan_filter_portion_max_{args.gan_filter_portion_max}' \
                   f'_gan_filter_portion_min_{args.gan_filter_portion_min}'
    if args.dataset_keep_portion < 1.0:
        log_dir += f'_dataset_keep_portion_{args.dataset_keep_portion}'
    os.makedirs(log_dir, exist_ok=True)

    gen_kwargs = {}

    if model_type == 'gan':
        train_data = gan_utils.get_train_data()
        train_fn = gan_utils.train
        train_with_teacher_fn = gan_utils.train_with_teacher
        gen_kwargs = {
            'gan_gen_filter': True,
            'gan_filter_portion_max': args.gan_filter_portion_max,
            'gan_filter_portion_min': args.gan_filter_portion_min,
        }
        gen_fn = gan_utils.gen_data
        get_model_assets_next_iter = gan_utils.get_model_assets
        get_train_data_next_iter = gan_utils.get_train_data_next_iter
        save_sample_fn = gan_utils.save_sample

    if model_type == 'vae':
        train_data = vae_utils.get_train_data()
        train_fn = vae_utils.train
        train_with_teacher_fn = vae_utils.train_with_teacher
        gen_fn = vae_utils.gen_data
        get_model_assets_next_iter = vae_utils.get_model_assets
        get_train_data_next_iter = vae_utils.get_train_data_next_iter
        save_sample_fn = vae_utils.save_sample

    if model_type == 'wta':
        train_data = wta_utils.get_train_data()
        train_fn = wta_utils.train
        train_with_teacher_fn = wta_utils.train_with_teacher
        gen_fn = wta_utils.gen_data
        get_model_assets_next_iter = wta_utils.get_model_assets
        get_train_data_next_iter = wta_utils.get_train_data_next_iter
        save_sample_fn = wta_utils.save_sample

    model_assets = get_model_assets_next_iter()
    generative_iterated_learning(model_assets, train_data, train_fn, gen_fn, total_iterations)
