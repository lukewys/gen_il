import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from wta_utils import weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SEM will not work in digits that are not setting.
class SEM(nn.Module):
    # https://github.com/iwyoo/tf_ConvWTA/blob/master/model.py
    def __init__(self):
        super(SEM, self).__init__()
        sz = 64
        self.sz = sz
        self.code_sz = 128

        self.enc = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(1, sz, 5, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(sz, sz, 5, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(sz, self.code_sz, 5, 1, padding=0),
            nn.ReLU(True),
        )
        self.sig = nn.Sigmoid()
        self.dec = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.code_sz, sz, 5, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(sz, sz, 5, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(sz, 1, 5, 1, 0)
        )

        self.tau = 0.5  # temperature

    def encode(self, x):
        h = self.enc(x.view(-1, 1, 28, 28))
        return h

    def decode(self, z):
        return self.dec(z)

    def sem(self, z):
        ori_shape = z.shape
        z = z.reshape(z.shape[0], z.shape[1], -1)
        z = F.softmax(z / self.tau, -1).reshape(ori_shape)

        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.sem(z)
        out = self.decode(z)
        return out


def get_new_model():
    model = SEM().to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


FIX_MODEL_INIT = None


def get_model_assets(model_assets=None, reset_model=True, use_same_init=True):
    global FIX_MODEL_INIT
    if reset_model:
        if use_same_init and FIX_MODEL_INIT is not None:
            if FIX_MODEL_INIT is None:
                FIX_MODEL_INIT = get_new_model()
            return copy.deepcopy(FIX_MODEL_INIT)
        else:
            return get_new_model()
    else:
        return model_assets
