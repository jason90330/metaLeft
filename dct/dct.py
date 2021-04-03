import torch
import math


def blockify(im, size):
    # bs = im.shape[0]
    ch = im.shape[0]
    h = im.shape[1]
    w = im.shape[2]

    im = im.view(ch, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=size, stride=size)
    im = im.transpose(1, 2)
    im = im.view(ch, -1, size, size)

    return im


def deblockify(blocks, ch, size):
    bs = blocks.shape[0] // ch
    block_size = blocks.shape[2]

    blocks = blocks.reshape(bs * ch, -1, block_size**2)
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=block_size, stride=block_size)
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks


def normalize(N):
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return (n @ n.t())


def harmonics(N):
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(im, device=None):
    N = im.shape[3]

    n = normalize(N)
    h = harmonics(N)

    if device is not None:
        n = n.to(device)
        h = h.to(device)

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ im @ h)

    return coeff


def block_idct(coeff, device=None):
    N = coeff.shape[3]

    n = normalize(N)
    h = harmonics(N)

    if device is not None:
        n = n.to(device)
        h = h.to(device)

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def batch_dct(im, device=None):
    ch = im.shape[0]
    size = (im.shape[1], im.shape[2])

    im_blocks = blockify(im, 8)
    dct_blocks = block_dct(im_blocks, device=device)
    dct = deblockify(dct_blocks, ch, size)

    return dct


def batch_idct(dct, device=None):
    ch = dct.shape[1]
    size = (dct.shape[2], dct.shape[3])

    dct_blocks = blockify(dct, 8)
    im_blocks = block_idct(dct_blocks, device=device)
    im = deblockify(im_blocks, ch, size)

    return im


def to_ycbcr(x, device=None):
    ycbcr_from_rgb = torch.Tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ]).view(3, 3).transpose(0, 1).to(device)

    # b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)
    b = torch.Tensor([0, 128, 128]).to(device).view(3, 1, 1).to(device)

    x = torch.einsum('cv,cxy->vxy', [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()


def to_rgb(x, device=None):
    rgb_from_ycbcr = torch.Tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ]).view(3, 3).transpose(0, 1).to(device)

    b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1).to(device)

    x -= b
    x = torch.einsum('cv,bcxy->bvxy', [rgb_from_ycbcr, x])

    return x.contiguous()


def prepare_dct(dct, stats, device=None, type=None):
    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

        t = ['y', 'cb', 'cr'][i] if type is None else type
        dct_blocks = stats.forward(dct_blocks, device=device, type=t)

        ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

    return torch.cat(ch, dim=1)


def unprepare_dct(dct, stats, device=None, type=None):
    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

        t = ['y', 'cb', 'cr'][i] if type is None else type
        dct_blocks = stats.backward(dct_blocks, device=device, type=t)

        ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

    return torch.cat(ch, dim=1)


def batch_to_images(dct, stats, device=None, scale_freq=True, crop=None, type=None):
    if scale_freq:
        dct = unprepare_dct(dct, stats, device=device, type=type)

    spatial = batch_idct(dct, device=device) + 128

    if spatial.shape[1] == 3:
        spatial = to_rgb(spatial, device)

    spatial = spatial.clamp(0, 255)
    spatial = spatial / 255

    if crop is not None:
        while len(crop.shape) > 1:
            crop = crop[0]

        cropY = crop[-2]
        cropX = crop[-1]

        spatial = spatial[:, :, :cropY, :cropX]

    return spatial


def images_to_batch(spatial, stats, device=None, type=None):
    spatial *= 255

    if spatial.shape[1] == 3:
        spatial = to_ycbcr(spatial, device)

    spatial -= 128

    frequency = batch_dct(spatial, device=device)
    return prepare_dct(frequency, stats, device=device, type=type)
