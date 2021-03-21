"""!
@brief SNR losses efficient computation in pytorch.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools

class PermInvariantSDR(nn.Module):
    """!
    Class for SDR computation between reconstructed signals and
    target wavs."""

    def __init__(self,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards
        """
        super().__init__()
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.n_sources = n_sources
        self.return_individual_results = return_individual_results

    def normalize_input(self, pr_batch, t_batch):
        min_len = min(pr_batch.shape[-1],
                      t_batch.shape[-1])
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(
                pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(
                t_batch, dim=-1, keepdim=True)
        return pr_batch, t_batch

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_snrs(self,
                              permuted_pr_batch,
                              t_batch,
                              eps=1e-9):
        s_t = t_batch
        e_t = permuted_pr_batch - s_t
        snrs = 10. * torch.log10((self.dot(s_t, s_t) + eps) /
                                 (self.dot(e_t, e_t) + eps))
        return snrs

    def compute_snr(self,
                    pr_batch,
                    t_batch,
                    eps=1e-9):

        initial_mixture = torch.sum(t_batch, -2, keepdim=True)

        snr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            snr = self.compute_permuted_snrs(
                permuted_pr_batch, t_batch, eps=eps)
            snr_l.append(snr)
        all_snrs = torch.cat(snr_l, -1)
        best_snr, best_perm_ind = torch.max(all_snrs.mean(-2), -1)

        if not self.return_individual_results:
            best_snr = best_snr.mean()

        if self.backward_loss:
            return -best_snr, best_perm_ind
        return best_snr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                return_best_permutation=False):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param eps: Numerical stability constant.

        :returns return_best_permutation Return the best permutation matrix
        """
        pr_batch, t_batch = self.normalize_input(
            pr_batch, t_batch)

        snr_l, best_perm_ind = self.compute_snr(
            pr_batch, t_batch, eps=eps)

        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return snr_l, best_permutations
        else:
            return snr_l

def test_snr_implementation():
    bs = 3
    n_sources = 4
    n_samples = 5
    zero_mean = False

    snr_loss = PermInvariantSDR(n_sources=n_sources,
                                zero_mean=zero_mean,
                                backward_loss=False)

    pr_batch = torch.ones((bs, n_sources, n_samples), dtype=torch.float32)
    t_batch = torch.ones((bs, n_sources, n_samples), dtype=torch.float32)

    pr_batch[:, 0:2, :] = 2.
    t_batch[:, 1:-1, :] = 2.

    snr_result, best_permutation = snr_loss(pr_batch, t_batch,
                                            eps=1e-9,
                                            return_best_permutation=True)
    print(snr_result, best_permutation)


def cross_check_with_museval():
    import museval

    bs = 1
    n_sources = 4
    n_secs = 3
    n_samples = n_secs * 44100
    zero_mean = False

    pr_batch = torch.rand((bs, n_sources, n_samples), dtype=torch.float32)
    t_batch = torch.rand((bs, n_sources, n_samples), dtype=torch.float32)

    print(pr_batch.mean(-1))
    print(pr_batch.std(-1))
    # pr_batch[:, 0:2, :] = pr_batch[:, 0:2, :]
    t_batch[:, -1, :] = 0.000000001 + 0.5
    # t_batch[:, -1, -1000:] = 1.
        # 1. + pr_batch[:, 0, :]

    snr_loss = PermInvariantSDR(n_sources=n_sources,
                                zero_mean=zero_mean,
                                backward_loss=False,
                                return_individual_results=True)

    snr_result = snr_loss(pr_batch, t_batch,
                          eps=1e-9, return_best_permutation=False)
    print(snr_result)

    print('After normalization')
    snr_result = snr_loss(
        (pr_batch - torch.mean(pr_batch, -1, keepdim=True)) / (
                torch.std(pr_batch, -1, keepdim=True) + 1e-9),
        (t_batch - torch.mean(t_batch, -1, keepdim=True)) / (
                torch.std(t_batch, -1, keepdim=True) + 1e-9),
        eps=1e-9, return_best_permutation=False)
    print(snr_result)

    museval_sdr, _, _, _, _ = museval.metrics.bss_eval(
        t_batch.permute(1, 2, 0).numpy(),
        pr_batch.permute(1, 2, 0).numpy(),
        compute_permutation=True,
        window=3*44100,
        hop=3*44100,
        framewise_filters=False,
        bsseval_sources_version=False
    )
    print(museval_sdr)
    print(museval_sdr.mean())


if __name__ == "__main__":
    cross_check_with_museval()
    # test_snr_implementation()