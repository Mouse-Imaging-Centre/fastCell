# https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-831-lb
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

from fastai.vision.learner import create_head, cnn_config, num_features_model, create_head
from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock


class Hcolumns(nn.Module):
    def __init__(self, hooks: Collection[Hook], nc: Collection[int] = None):
        super(Hcolumns, self).__init__()
        self.hooks = hooks
        self.n = len(self.hooks)
        self.factorization = None
        if nc is not None:
            self.factorization = nn.ModuleList()
            for i in range(self.n):
                self.factorization.append(nn.Sequential(
                    conv2d(nc[i], nc[-1], 3, padding=1, bias=True),
                    conv2d(nc[-1], nc[-1], 3, padding=1, bias=True)))
                # self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))

    def forward(self, x: Tensor):
        n = len(self.hooks)
        out = [F.interpolate(self.hooks[i].stored if self.factorization is None
                             else self.factorization[i](self.hooks[i].stored), scale_factor=2 ** (self.n - i),
                             mode='bilinear', align_corners=False) for i in range(self.n)] + [x]
        return torch.cat(out, dim=1)


class DynamicUnet_Hcolumns(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(self, encoder: nn.Module, n_classes: int, blur: bool = False, blur_final=True,
                 self_attention: bool = False,
                 y_range: Optional[Tuple[float, float]] = None,
                 last_cross: bool = True, bottle: bool = False, **kwargs):
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni * 2, **kwargs),
                                    conv_layer(ni * 2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
        hc_c = [x.shape[1]]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
                                   blur=blur, self_attention=sa, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)
            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        hc_c.append(ni)
        layers.append(Hcolumns(self.hc_hooks, hc_c))
        layers += [conv_layer(ni * len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()


def unet_learner(data: DataBunch, arch: Callable, pretrained: bool = True, blur_final: bool = True,
                 norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                 blur: bool = False, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                 last_cross: bool = True, bottle: bool = False, cut=None,
                 hypercolumns=True, **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    M = DynamicUnet_Hcolumns if hypercolumns else DynamicUnet
    model = to_device(M(body, n_classes=data.c, blur=blur, blur_final=blur_final,
                        self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                        last_cross=last_cross, bottle=bottle), data.device)
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn