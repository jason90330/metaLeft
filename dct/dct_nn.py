from .dct import blockify, deblockify
import torch
import math

# box resize, takes and 8 x 8 image and returns a 16 x 16 image
def double_size_tensor():
    op = torch.zeros((8, 8, 16, 16))
    for i in range(0, 8):
        for j in range(0, 8):
            for u in range(0, 16):
                for v in range(0, 16):
                    if i == u // 2 and j == v // 2:
                        op[i, j, u, v] = 1

    return op


# box resize, takes a 16 x 16 and returns an 8 x 8
def half_size_tensor():
    op = torch.zeros((16, 16, 8, 8))
    for i in range(0, 16):
        for j in range(0, 16):
            for u in range(0, 8):
                for v in range(0, 8):
                    if i == 2*u and j == 2*v:
                        op[i, j, u, v] = 1

    return op


# DCT takes 8 x 8 pixels and returns 8 x 8 coefficients
def A(alpha):
    if alpha == 0:
        return 1.0 / math.sqrt(2)
    else:
        return 1


def D():
    D_t = torch.zeros((8, 8, 8, 8))
    
    for i in range(8):
        for j in range(8):
            for alpha in range(8):
                for beta in range(8):
                    scale_a = A(alpha)
                    scale_b = A(beta)
                    
                    coeff_x = math.cos(((2 * i + 1) * alpha * math.pi) / 16)
                    coeff_y = math.cos(((2 * j + 1) * beta * math.pi) / 16)
                    
                    D_t[i, j, alpha, beta] = 0.25 * scale_a * scale_b * coeff_x * coeff_y
    return D_t

# reblocker, takes a 16 x 16 and returns 4 8 x 8 blocks
def reblock():
    blocks_shape = (2, 2)
    B_t = torch.zeros((16, 16, 4, 8, 8))
    
    for s_x in range(16):
        for s_y in range(16):
            for n in range(4):
                for i in range(8):
                    for j in range(8):
                        x = n % 2
                        y = n // 2
                        if x * 8 + i == s_x and y * 8 + j == s_y:
                            B_t[s_x, s_y, n, i, j] = 1.0

    return B_t


# takes 4 x 8 x 8 and returns a 16 x 16
def macroblock():
    blocks_shape = (2, 2)
    B_t = torch.zeros((4, 8, 8, 16, 16))
    
    # 0 goes in top left
    for alpha in range(8):
        for beta in range(8):
            B_t[0, alpha, beta, alpha, beta] = 1

    # 1 goes in top right
    for alpha in range(8):
        for beta in range(8):
            B_t[1, alpha, beta, alpha + 8, beta] = 1

    # 2 goes in bottom left
    for alpha in range(8):
        for beta in range(8):
            B_t[2, alpha, beta, alpha, beta + 8] = 1

    # 3 goes in bottom right
    for alpha in range(8):
        for beta in range(8):
            B_t[3, alpha, beta, alpha + 8, beta + 8] = 1
                    
    return B_t


resizer = double_size_tensor()
halfsizer = half_size_tensor()
dct = D()
reblocker = reblock()
macroblocker = macroblock()

# block doubler combines the following linear operations in order: inverse DCT, NN doubling, reshape to 4 x 8 x 8, DCT, reshape back to 16 x 16
block_doubler = torch.einsum('ijab,ijmn,mnzxy,xypq,zpqrw->abrw', dct, resizer, reblocker, dct, macroblocker)

# 16 x 16 -> 4 x 8 x 8 -> idct -> 16 x 16 -> resize -> dct
block_halver = torch.einsum('mnzab,ijab,zijrw,rwxy,xypq->mnpq', reblocker, dct, macroblocker, halfsizer, dct)


def double_nn_dct(input_dct, device=None):
    if device is not None:
        dop = block_doubler.cuda()
    else:
        dop = block_doubler

    dct_blocks = blockify(input_dct, 8)
    dct_doubled = torch.einsum('abrw,cdab->cdrw', [dop, dct_blocks])
    deblocked_doubled = deblockify(dct_doubled, input_dct.shape[1], (input_dct.shape[2] * 2, input_dct.shape[3] * 2))

    return deblocked_doubled


def half_nn_dct(input_dct, device=None):
    if device is not None:
        dop = block_halver.cuda()
    else:
        dop = block_halver

    dct_blocks = blockify(input_dct, 16)
    dct_halved  = torch.einsum('abrw,cdab->cdrw', [dop, dct_blocks])
    deblocked_halved = deblockify(dct_halved, input_dct.shape[1], (input_dct.shape[2] // 2, input_dct.shape[3] // 2))

    return deblocked_halved