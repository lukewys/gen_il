"""Iterative reconstruction utilities for digits that are not."""

import torch


def renormalize(x, neg_1_to_1=False):
    # TODO: move the judge of the renorm type inside this function.
    """
    Re-normalize a batch of images to be between [0,1] or [-1,1].
    """
    ch = x.shape[-3]
    w = x.shape[-2]
    h = x.shape[-1]
    x = x.view(-1, ch, w * h)
    x = (x - torch.min(x, dim=-1, keepdim=True)[0]) / (
            torch.max(x, dim=-1, keepdim=True)[0] - torch.min(x, dim=-1, keepdim=True)[0])

    # normalize to -1, 1
    if neg_1_to_1:
        x = x * 2 - 1
    return x.view(-1, ch, w, h)


def recon_till_converge(model, recon_fn, image_batch,
                        thres=1e-6, return_history=False,
                        max_iteration=100, renorm='none'):
    """
    Reconstructs a batch of images until the reconstruction error is below a threshold.
    image_batch: [batch, ch, w, h]
    renorm: 'none', '0_to_1', '-1_to_1'
    """
    diff = 1e10
    history = [image_batch]
    model.eval()
    iteration = 1
    with torch.no_grad():
        while diff > thres:
            recon_batch = recon_fn(model, image_batch)
            diff = torch.max(torch.mean((image_batch - recon_batch).pow(2), dim=(2, 3)))
            if return_history:
                history.append(recon_batch)
            # re-normalize image
            if renorm == '0_to_1':
                image_batch = renormalize(recon_batch)
            elif renorm == '-1_to_1':
                image_batch = renormalize(recon_batch, neg_1_to_1=True)
            elif renorm == 'none' or renorm is None:
                image_batch = recon_batch
            else:
                raise ValueError(f'Unknown renorm type: {renorm}')
            iteration += 1
            if iteration > max_iteration:
                break
    if return_history:
        return image_batch, history
    else:
        return image_batch
