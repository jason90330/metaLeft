import torch
from .stats import Stats
from .quantization import quantization_max


class QuantizationStats(Stats):
    def __init__(self, root, type='ms'):
        super(QuantizationStats, self).__init__(root, type)

        if root is not None:
            self.mean = {
                'luma': self.stats['mean_luma'].view(8, 8),
                'chroma': self.stats['mean_chroma'].view(8, 8),
            }

            self.variance = {
                'luma': self.stats['variance_luma'].view(8, 8),
                'chroma': self.stats['variance_chroma'].view(8, 8),
            }

            self.std = {t: torch.sqrt(v) for t, v in self.variance.items()}

    def _mean_variance_f(self, blocks, device=None, table='luma'):
        return self.__normalize(
            self.__center(
                blocks, device, table
            ), device, table
        )

    def _zero_one_f(self, blocks, device=None, table='luma'):
        return blocks / quantization_max

    def _mean_variance_r(self, blocks, device=None, table='luma'):
        return self.__uncenter(
            self.__denormalize(
                blocks, device, table
            ), device, table
        )

    def _zero_one_r(self, blocks, device=None, table='luma'):
        return blocks * quantization_max

    def __center(self, blocks, device=None, table='luma'):
        m = self.mean[table]

        if device is not None:
            m = m.to(device)

        return blocks - m

    def __uncenter(self, blocks, device=None, table='luma'):
        m = self.mean[table]

        if device is not None:
            m = m.to(device)

        return blocks + m

    def __normalize(self, blocks, device=None, table='luma'):
        s = self.std[table]

        if device is not None:
            s = s.to(device)

        return blocks / s

    def __denormalize(self, blocks, device=None, table='luma'):
        s = self.std[table]

        if device is not None:
            s = s.to(device)

        return blocks * s
