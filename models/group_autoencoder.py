import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from vit_pytorch.vit import Transformer
from models.group_vit import GroupViT, get_attn_maps
from einops import rearrange


class GroupDecoder(nn.Module):
    def __init__(
            self,
            *,
            pixel_values_per_patch=16 * 16 * 3,
            num_patches=14 * 14,
            encoder_dim=384,
            decoder_dim=384,
            decoder_depth=4,
            decoder_heads=8,
            decoder_dim_head=64
    ):
        super().__init__()

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, num_patches, self.decoder_dim))
        trunc_normal_(self.decoder_pos_emb, std=.02)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, encoded_tokens):
        device = encoded_tokens.device

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # add position embedding
        decoder_tokens = decoder_tokens + self.decoder_pos_emb

        decoded_tokens = self.decoder(decoder_tokens)
        # splice out the mask tokens and project to pixel values
        pred_pixel_values = self.to_pixels(decoded_tokens)

        # calculate reconstruction loss

        # recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return pred_pixel_values


class GroupAutoencoder(nn.Module):
    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_chans=3,

    ):
        super().__init__()
        self.num_patch_h = image_size // patch_size
        self.num_patch_w = image_size // patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = GroupViT(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=384,
            embed_factors=[1, 1, 1],
            depths=[6, 3, 3],
            num_heads=[6, 6, 6],
            num_group_tokens=[32, 16, 0],
            num_output_groups=[32, 16],
        )
        self.decoder = GroupDecoder(
            pixel_values_per_patch=patch_size * patch_size * in_chans,
            num_patches=self.num_patches,
            encoder_dim=384,
            decoder_dim=384,
            decoder_depth=4,
            decoder_heads=8,
            decoder_dim_head=64
        )
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, img):
        out = self.encoder(img, return_feat=True, return_attn=True, as_dict=True)
        attn_maps = get_attn_maps(out, return_onehot=True, rescale=False, num_patch_h=self.num_patch_h,
                                  num_patch_w=self.num_patch_w)
        # out['feat']: [B, N, C]
        # attn_maps[-1]: [B, H, W, G]
        # [B, N, C], [B, H, W, G] -> [B, H, W, C]
        group_tokens_tiled = torch.einsum('bnc,bhwg->bhwc', out['feat'], attn_maps[-1])
        # [B, H, W, C] -> [B, H*W, C]
        encoded_tokens = rearrange(group_tokens_tiled, 'b h w c -> b (h w) c')
        img_recon = self.decoder(encoded_tokens)
        # img_recon: [B, H*W, 3*16*16] -> [B, 3, H*16, W*16]
        img_recon = rearrange(img_recon, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                              p1=self.patch_size, p2=self.patch_size,
                              h=group_tokens_tiled.shape[1],
                              w=group_tokens_tiled.shape[2])
        return img_recon


from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import pair


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim,
                 depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        return x


class ViTAutoencoder(nn.Module):
    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_chans=3,

    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patch_h = image_size // patch_size
        self.num_patch_w = image_size // patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            channels=in_chans,
            dim=384,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.encoder.mlp_head = nn.Identity()  # remove the mlp head
        self.decoder = GroupDecoder(
            pixel_values_per_patch=patch_size * patch_size * in_chans,
            num_patches=self.num_patches,
            encoder_dim=384,
            decoder_dim=384,
            decoder_depth=4,
            decoder_heads=8,
            decoder_dim_head=64
        )

    def forward(self, img):
        # [B, C, H, W] -> [B, H*W, C]
        encoded_tokens = self.encoder(img)
        img_recon = self.decoder(encoded_tokens)
        # img_recon: [B, H*W, 3*16*16] -> [B, 3, H*16, W*16]
        img_recon = rearrange(img_recon, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                              p1=self.patch_size, p2=self.patch_size,
                              h=self.num_patch_h,
                              w=self.num_patch_w)
        return img_recon


from diffusers.models.vae import Decoder as VAEDecoder


class ViTCNNAutoencoder(nn.Module):
    def __init__(self,
                 image_size=64,
                 patch_size=4,
                 in_chans=3, ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patch_h = image_size // patch_size
        self.num_patch_w = image_size // patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            channels=in_chans,
            dim=384,
            depth=4,
            heads=8,
            mlp_dim=384 * 4,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.decoder = VAEDecoder(
            in_channels=384,
            out_channels=in_chans,
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            block_out_channels=[
                64,
                128,
                128,
            ],
            layers_per_block=2, )

    def forward(self, img):
        # [B, C, H, W] -> [B, H*W, C]
        encoded_tokens = self.encoder(img)
        # [B, H*W, C] -> [B, C, H, W]
        encoded_tokens = rearrange(encoded_tokens, 'b (h w) c -> b c h w', h=self.num_patch_h, w=self.num_patch_w)
        img_recon = self.decoder(encoded_tokens)
        return img_recon


class GroupCNNAutoencoder(nn.Module):
    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_chans=3,
    ):
        super().__init__()
        self.num_patch_h = image_size // patch_size
        self.num_patch_w = image_size // patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = GroupViT(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=384,
            embed_factors=[1, 1],
            depths=[6, 3],
            num_heads=[6, 6],
            num_group_tokens=[16, 0],
            num_output_groups=[16],
            gumbel_taus=[1.0],
            hard_assignment=True,
        )
        self.decoder = VAEDecoder(
            in_channels=384,
            out_channels=in_chans,
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            block_out_channels=[
                64,
                128,
                128,
            ],
            layers_per_block=2, )
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, img):
        out = self.encoder(img, return_feat=True, return_attn=True, as_dict=True)
        attn_maps = get_attn_maps(out, return_onehot=True, rescale=False, num_patch_h=self.num_patch_h,
                                  num_patch_w=self.num_patch_w)
        # out['feat']: [B, G, C]
        # attn_maps[-1]: [B, H, W, G]
        # [B, G, C], [B, H, W, G] -> [B, H, W, C]
        group_tokens_tiled = torch.einsum('bgc,bhwg->bhwc', out['feat'], attn_maps[-1])
        # [B, H, W, C] -> [B, C, H, W]
        group_tokens_tiled = rearrange(group_tokens_tiled, 'b h w c -> b c h w')
        img_recon = self.decoder(group_tokens_tiled)
        return img_recon

    def get_colored_attn_map(self, img, color_map, layer=-1):
        out = self.encoder(img, return_feat=True, return_attn=True, as_dict=True)
        attn_maps = get_attn_maps(out, return_onehot=True, rescale=False, num_patch_h=self.num_patch_h,
                                  num_patch_w=self.num_patch_w)
        # color_map: [G, C]
        # attn_maps[-1]: [B, H, W, G]
        # [G, C], [B, H, W, G] -> [B, H, W, C]
        attn_map_colored = torch.einsum('gc,bhwg->bhwc', color_map, attn_maps[layer])
        # [B, H, W, C] -> [B, C, H, W]
        attn_map_colored = rearrange(attn_map_colored, 'b h w c -> b c h w')
        return attn_map_colored

# TODO: entropy regulariztaion loss
# 注意: 拿out['attn_dicts'][0]['soft']. shape是[128, 1, 32, 256] ([Batch, num_heads, num_group_tokens, num_image_tokens).
# softmax是在-2维做的。
def entropy_regularization(attention_map, dim=-1):
    """
    Computes the entropy regularization loss for an attention map
    """
    batch, N, S = attention_map.shape
    attention_map = torch.softmax(attention_map, dim=dim)
    log_probs = torch.log(attention_map + 1e-10)
    entropy = -torch.sum(attention_map * log_probs, dim=dim)
    entropy = torch.mean(entropy)
    return entropy


# https://github.com/NVlabs/GroupViT/blob/main/utils/optimizer.py
from torch import optim as optim


def build_optimizer(model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    parameters = set_weight_decay(model, {}, {})

    optimizer = optim.AdamW(
        parameters,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=1.6e-3,
        weight_decay=0.05)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


# https://github.com/NVlabs/GroupViT/blob/main/utils/lr_scheduler.py
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(epochs, optimizer, n_iter_per_epoch):
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(2 * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        t_mul=1.,
        lr_min=4e-5,
        warmup_lr_init=4e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler


if __name__ == '__main__':
    decoder = VAEDecoder(
        in_channels=4,
        out_channels=3,
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ],
        block_out_channels=[
            128,
            256,
            512,
            512
        ],
        layers_per_block=2, )

    x = torch.randn(2, 4, 16, 16)

    out = decoder(x)
    print(out.shape)

    # model = GroupAutoencoder()
    # x = torch.randn(2, 3, 224, 224)
    # out = model(x)
    # print(out)
