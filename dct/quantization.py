import torch
from .dct import blockify, deblockify, batch_dct, batch_idct


luma_quant_matrix = torch.Tensor([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
])

chroma_quant_matrix = torch.Tensor([
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
])

quantization_max = 255.


def quality_to_scale_factor(quality):
    quality = int(quality)

    if quality <= 0:
        quality = 1

    if quality > 100:
        quality = 100

    if quality < 50:
        quality = 5000 // quality
    else:
        quality = int(200 - quality * 2)

    return quality


def scale_quantization_matrix(scale_factor, table='luma'):
    if table == 'luma':
        t = luma_quant_matrix
    elif table == 'chroma':
        t = chroma_quant_matrix

    mat = (t * scale_factor + 50) // 100
    mat[mat <= 0] = 1
    mat[mat > 255] = 255

    return mat


def get_coefficients_for_quality(quality, table='luma'):
    scaler = quality_to_scale_factor(quality)
    mat = scale_quantization_matrix(scaler, table=table)
    return mat.reshape(8, 8)


def quantize(dct, mat):
    dct_blocks = blockify(dct, 8)
    quantized_blocks = torch.round(dct_blocks / mat)
    quantized = deblockify(quantized_blocks, dct.shape[1], (dct.shape[2], dct.shape[3]))
    return quantized


def dequantize(dct, mat):
    dct_blocks = blockify(dct, 8)
    dequantized_blocks = dct_blocks * mat
    dequantized = deblockify(dequantized_blocks, dct.shape[1], (dct.shape[2], dct.shape[3]))
    return dequantized


def quantize_at_quality(dct_blocks, quality, table='luma'):
    mat = get_coefficients_for_quality(quality, table=table)
    return quantize(dct_blocks, mat)


def dequantize_at_quality(dct_blocks, quality, table='luma'):
    mat = get_coefficients_for_quality(quality, table=table)
    return dequantize(dct_blocks, mat)


def compress_at_quality(batch, quality, table='luma'):
    batch = batch * 255 - 128
    d = batch_dct(batch)
    d = quantize_at_quality(d, quality, table=table)
    return d


def decompress_at_quality(batch, quality, table='luma'):
    d = dequantize_at_quality(batch, quality, table=table)
    d = batch_idct(d)
    d = (d + 128) / 255
    return d
