

from .guided_ddpm_plain_unet import GuidedDDPMPlainUNet


def build_network(cfg):
    attention_ds = []
    for res in cfg.attention_resolutions.split(","):
        if res == "":
            continue
        attention_ds.append(cfg.image_size // int(res))

    if cfg.channel_mult == "":
        if cfg.large_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif cfg.large_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:
            raise ValueError
    else:
        channel_mult = tuple(int(ch_mult)
                             for ch_mult in cfg.channel_mult.split(","))

    if cfg.image_type == "rgb":
        image_channels = 3
    elif cfg.image_type == "mrc":
        image_channels = 1
    else:
        raise ValueError
    return GuidedDDPMPlainUNet(
        image_size=cfg.image_size,  # useless
        in_channels=image_channels,
        model_channels=cfg.model_channels,
        out_channels=image_channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=cfg.dropout,
        channel_mult=channel_mult,
        conv_resample=True,
        dims=2,
        use_checkpoint=cfg.use_checkpoint,
        num_heads=cfg.num_heads,
        num_head_channels=cfg.num_head_channels,
        num_heads_upsample=cfg.num_heads_upsample,
        resblock_updown=cfg.resblock_updown,
        use_new_attention_order=cfg.use_new_attention_order,
    )
