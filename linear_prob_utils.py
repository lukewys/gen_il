import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_utils import LARS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):
    # https://github.com/facebookresearch/moco/blob/main/main_lincls.py#L434
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # (yusong) change to only print the average value
        fmtstr = 'Avg {name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, base_lr, epoch, milestones):
    """Decay the learning rate based on schedule"""
    lr = base_lr
    for milestone in milestones:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LinearProbeModel(nn.Module):
    def __init__(self, enc_model, input_dim, output_dim, get_latent_fn):
        super(LinearProbeModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.enc_model = enc_model

        self.get_latent_fn = get_latent_fn

        # init linear layer
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def get_trainable_parameters(self):
        return self.linear.parameters()

    def forward(self, x):
        h = self.get_latent_fn(self.enc_model, x)
        return self.linear(h)


def linear_probe(model, get_data_fn, batch_size=8192, num_epochs=50):
    # https://github.com/facebookresearch/moco/blob/main/main_lincls.py#L308
    # (yusong) the optimizer hyperparameter is taken from the link, but the lr is set by myself.

    train_loader, test_loader = get_data_fn(batch_size)

    base_lr = 1
    milestones = [30, 40]

    model.eval()
    optimizer = optim.SGD(model.get_trainable_parameters(), lr=base_lr, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0
    for epoch in range(num_epochs):
        # train one epoch
        end = time.time()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        adjust_learning_rate(optimizer, epoch, base_lr, milestones)

        entries = [f'Train Epoch: {epoch}'] + [str(meter) for meter in [batch_time, losses, top1, top5]]
        print('\t'.join(entries))

        # evaluate on test set
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            for images, target in test_loader:
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

        entries = [f'Test Epoch: {epoch}'] + [str(meter) for meter in [losses, top1, top5]]
        print('\t'.join(entries))

        if top1.avg > max_acc:
            max_acc = top1.avg

    return max_acc
