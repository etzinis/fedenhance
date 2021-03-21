"""!
@brief Loss computed under some norm for the estimation of a signal or a mask.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools


class L1(nn.Module):
    """!
    Class for estimation under norm computation between estimated signals
    and target signals."""

    def __init__(self,
                 return_individual_results=False):
        super().__init__()
        self.return_individual_results = return_individual_results

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-8):
        """!

        :param pr_batch: Estimated signals: Torch Tensors of size:
                         batch_size x ...
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x ...
        :param eps: Numerical stability constant

        :returns normed loss for both forward and backward passes.
        """
        error = torch.abs(pr_batch - t_batch)
        l1_loss = torch.mean(error.view(error.shape[0], -1), -1)

        if not self.return_individual_results:
            l1_loss = l1_loss.mean()
        return l1_loss


class PermInvariantL1(nn.Module):
    """!
    Class for estimation under norm computation between estimated signals
    and target signals."""

    def __init__(self,
                 n_sources=2,
                 weighted_norm=False,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.weighted_norm = weighted_norm
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.return_individual_results = return_individual_results

    def compute_perm_invariant_loss(self, pr_batch, t_batch):
        perm_l = []
        for perm in self.permutations:
            permuted_pr_batch = (pr_batch[:, perm, :])
            error = torch.abs(permuted_pr_batch - t_batch)
            if self.weighted_norm:
                error *= (t_batch ** 2)
            l1 = torch.mean(error.view(error.shape[0], -1), dim=1)
            perm_l.append(l1)

        all_losses = torch.stack(perm_l, dim=1)
        best_loss, best_perm_ind = torch.min(all_losses.mean(-2), -1)

        if not self.return_individual_results:
            best_loss = best_loss.mean()

        return best_loss, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-8,
                return_best_permutation=False):
        """!

        :param pr_batch: Estimated signals: Torch Tensors of size:
                         batch_size x ...
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x ...
        :param eps: Numerical stability constant

        :returns normed loss for both forward and backward passes.
        """
        best_loss, best_perm_ind = self.compute_perm_invariant_loss(
            pr_batch, t_batch)

        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return best_loss, best_permutations
        else:
            return best_loss

