import torch
from .stats import Stats


class DCTStats(Stats):
    def __init__(self, root, type='ms'):
        super(DCTStats, self).__init__(root, type)
        stats = torch.load(root)

        if 'y' in stats:
            self.mean = {x: stats[x]['mean'].view(1, 1, 8, 8) for x in stats.keys()}
            self.variance = {x: stats[x]['variance'].view(1, 1, 8, 8) for x in stats.keys()}
            self.std = {x: torch.sqrt(self.variance[x]) for x in stats.keys()}

            self.min = {x: stats[x]['min'].view(1, 1, 8, 8) for x in stats.keys()}
            self.max = {x: stats[x]['max'].view(1, 1, 8, 8) for x in stats.keys()}
        else:
            self.mean = {'y': stats['mean'].view(1, 1, 8, 8)}
            self.variance = {'y': stats['variance'].view(1, 1, 8, 8)}
            self.std = {'y': torch.sqrt(self.variance['y'])}

            self.min = {'y': stats['min'].view(1, 1, 8, 8)}
            self.max = {'y': stats['max'].view(1, 1, 8, 8)}

    def _mean_variance_f(self, blocks, type='y', device=None):
        m = self.mean[type]

        if device is not None:
            m = m.to(device)

        blocks = blocks - m

        s = self.std[type]

        if device is not None:
            s = s.to(device)

        return blocks / s

    def _zero_one_f(self, blocks, type='y', device=None):
        m = -self.min[type]

        if device is not None:
            m = m.to(device)

        blocks = blocks + m

        s = self.max[type] - self.min[type]

        if device is not None:
            s = s.to(device)

        return blocks / s

    def _mean_variance_r(self, blocks, type='y', device=None):
        s = self.std[type]

        if device is not None:
            s = s.to(device)

        blocks = blocks * s

        m = self.mean[type]

        if device is not None:
            m = m.to(device)

        return blocks + m

    def _zero_one_r(self, blocks, type='y', device=None):
        s = self.max[type] - self.min[type]

        if device is not None:
            s = s.to(device)

        blocks = blocks * s

        m = -self.min[type]

        if device is not None:
            m = m.to(device)

        return blocks - m
