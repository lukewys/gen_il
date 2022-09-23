import os
import argparse
import numpy as np

from utils.train_utils import str2bool, set_seed
from utils.data_utils import get_init_data
from models import sem_utils, wta_utils, vae_utils, gan_utils
from evaluate.linear_probe import linear_probe


def train_model(model_assets, train_data, train_fn, iteration):
    print(f'=======Iteration {iteration}: Train Model=======')
    model_assets = train_fn(model_assets, train_data, train_extend)
    return model_assets


def eval_model(model_assets, test_data, eval_fn, iteration):
    print(f'=======Iteration {iteration}: Eval Model=======')
    eval_fn(model_assets, test_data, transform, log_dir=log_dir, iteration=iteration)


def train_model_with_teacher(new_model_assets, old_model_assets, train_with_teacher_fn, iteration, **gen_kwargs):
    print(f'=======Iteration {iteration}: Train Model With Teacher=======')
    total_steps = 3000
    total_steps = int(np.round(train_extend * total_steps))
    model_assets = train_with_teacher_fn(new_model_assets, old_model_assets, total_steps, **gen_kwargs)
    return model_assets


def generate_data(model_assets, gen_fn, iteration, **gen_kwargs):
    print(f'=======Iteration {iteration}: Generate Data=======')
    data_generated = gen_fn(model_assets, gen_batch_size, gen_num_batch, **gen_kwargs)
    # data_generated: tensor of [batch_size, *]
    return data_generated


def save_generated_data(model_assets, iteration, **gen_kwargs):
    # data_generated: tensor of [batch_size, *]
    save_sample_fn(model_assets, log_dir, iteration, transform, **gen_kwargs)


def generative_iterated_learning(model_assets, train_data, train_fn, gen_fn, total_iterations, gen_kwargs):
    for iteration in range(1, total_iterations + 1):
        if train_with_teacher and iteration > 1:
            model_assets = train_model_with_teacher(new_model_assets, model_assets, train_with_teacher_fn, iteration,
                                                    **gen_kwargs)
        else:
            model_assets = train_model(model_assets, train_data, train_fn, iteration)
            data_generated = generate_data(model_assets, gen_fn, iteration, **gen_kwargs)
            train_data = get_train_data_next_iter(train_data, data_generated, add_old_dataset=add_old_dataset,
                                                  keep_portion=dataset_keep_portion)
        eval_model(model_assets, test_data, eval_fn, iteration)

        if iteration % save_image_interval == 0:
            save_generated_data(model_assets, iteration, **gen_kwargs)

        if train_linear_probe:
            linear_probe_model = get_linear_probe_model_fn(model_assets)
            max_acc = linear_probe(linear_probe_model, dataset_name, model_utils.get_transform(dataset_name))
            with open(os.path.join(log_dir, 'max_acc.txt'), 'a') as f:
                f.write(f'Iter {iteration}: {max_acc}\n')

        print(f'=======Iteration {iteration}: Get New Model=======')
        new_model_assets = get_model_assets_next_iter(model_assets=model_assets, reset_model=reset_model,
                                                      use_same_init=use_same_init)
        if not train_with_teacher:
            model_assets = new_model_assets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gen IL')
    parser.add_argument('--model_type', type=str, default='wta', metavar='N',
                        help='model_type')
    parser.add_argument('--save_image_interval', type=int, default=1, metavar='N',
                        help='save_image_interval')
    parser.add_argument('--total_iterations', type=int, default=10, metavar='N',
                        help='total_iterations')
    parser.add_argument('--add_old_dataset', type=str2bool, nargs='?', const=True,
                        default=False, help='add old dataset')
    parser.add_argument('--train_extend', type=float, default=1.0, metavar='S',
                        help='train_extend')
    parser.add_argument('--gen_num_batch', type=int, default=120, metavar='N',
                        help='gen_num_batch')
    parser.add_argument('--train_with_teacher', type=str2bool, nargs='?', const=True,
                        default=False, help='train_with_teacher')
    parser.add_argument('--gan_filter_portion_max', type=float, default=None, metavar='S',
                        help='gan_filter_portion_max')
    parser.add_argument('--gan_filter_portion_min', type=float, default=None, metavar='S',
                        help='gan_filter_portion_min')
    parser.add_argument('--dataset_keep_portion', type=float, default=1.0, metavar='S',
                        help='dataset_keep_portion')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='seed')
    parser.add_argument('--reset_model', type=str2bool, nargs='?', const=True, default=True, help='reset_model')
    parser.add_argument('--use_same_init', type=str2bool, nargs='?', const=True, default=True, help='use_same_init')
    parser.add_argument('--linear_probe', type=str2bool, nargs='?', const=True, default=False, help='linear_probe')
    parser.add_argument('--holdout_digits', type=str, default=None, metavar='S', help='holdout_digits')
    parser.add_argument('--dataset_name', type=str, default='mnist', metavar='S', help='dataset name')
    parser.add_argument('--gen_norm', type=str, default='none', metavar='S', help='normalization when generating data')
    parser.add_argument('--model_kwargs', type=str, default='{}', metavar='N', help='model kwargs')
    args = parser.parse_args()

    model_kwargs = eval(args.model_kwargs)

    set_seed(args.seed)

    dataset_name = args.dataset_name
    save_image_interval = args.save_image_interval
    total_iterations = args.total_iterations
    train_extend = args.train_extend  # % of training steps
    add_old_dataset = args.add_old_dataset
    gen_batch_size = 500
    gen_num_batch = args.gen_num_batch  # for matching the size of mnist train data, =120
    model_type = args.model_type  # gan, vae, wta
    train_with_teacher = args.train_with_teacher
    dataset_keep_portion = args.dataset_keep_portion
    reset_model = args.reset_model
    use_same_init = args.use_same_init
    train_linear_probe = args.linear_probe
    if args.holdout_digits is not None:
        holdout_digits = eval(args.holdout_digits)
        assert type(holdout_digits) == list
    else:
        holdout_digits = None

    # print all the args and their values
    print('\n'.join(f'{k}: {v}' for k, v in sorted(vars(args).items())))

    log_dir = f'gen_il_logs/{model_type}_{dataset_name}_total_iter_{total_iterations}_train_extend_{train_extend}_' \
              f'gen_num_batch_{gen_num_batch}_add_old_dataset_{add_old_dataset}_' \
              f'train_with_teacher_{train_with_teacher}_seed_{args.seed}_reset_model_{reset_model}_' \
              f'use_same_init_{use_same_init}_train_linear_probe_{train_linear_probe}_' \
              f'holdout_digits_{holdout_digits}_gen_norm_{args.gen_norm}'

    if args.gan_filter_portion_max and args.gan_filter_portion_min:
        log_dir += f'_gan_filter_portion_max_{args.gan_filter_portion_max}' \
                   f'_gan_filter_portion_min_{args.gan_filter_portion_min}'
    if args.dataset_keep_portion < 1.0:
        log_dir += f'_dataset_keep_portion_{args.dataset_keep_portion}'

    if model_kwargs:
        log_dir += f'_model_kwargs_{args.model_kwargs}'
    os.makedirs(log_dir, exist_ok=True)

    gen_kwargs = {'renorm': args.gen_norm}

    if model_type == 'gan':
        model_utils = gan_utils
        gen_kwargs = {
            'gan_gen_filter': True,
            'gan_filter_portion_max': args.gan_filter_portion_max,
            'gan_filter_portion_min': args.gan_filter_portion_min,
        }
    if model_type == 'vae':
        model_utils = vae_utils
    elif model_type == 'wta':
        model_utils = wta_utils
    elif model_type == 'sem':
        model_utils = sem_utils
    else:
        raise NotImplementedError

    train_fn = model_utils.train
    eval_fn = model_utils.evaluate
    train_with_teacher_fn = model_utils.train_with_teacher
    gen_fn = model_utils.gen_data
    get_model_assets_next_iter = model_utils.get_model_assets
    get_train_data_next_iter = model_utils.get_train_data_next_iter
    save_sample_fn = model_utils.save_sample
    get_linear_probe_model_fn = model_utils.get_linear_probe_model
    transform = model_utils.get_transform(dataset_name)

    train_data, test_data = get_init_data(transform, dataset_name, model_utils.BATCH_SIZE)

    model_assets = get_model_assets_next_iter(reset_model=reset_model, use_same_init=use_same_init, **model_kwargs)
    generative_iterated_learning(model_assets, train_data, train_fn, gen_fn, total_iterations, gen_kwargs)
