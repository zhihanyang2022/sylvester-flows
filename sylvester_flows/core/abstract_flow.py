import torch.nn as nn


class AbstractFlow(nn.Module):

    def sample_and_compute_logp(self, num_samples: int):
        raise NotImplementedError

    def sample_from_each_layer(self, num_samples: int):
        raise NotImplementedError
