"""!
@brief SNR losses efficient computation in pytorch.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools
from torch.nn.modules.loss import _Loss


class FixedMixIT2Sources2NoisesSNRwithZeroRefs(nn.Module):
    """!
    Class for SNR computation between 4 reconstructed signals and
    2 reference speakers and 2 reference noises."""

    def __init__(self,
                 zero_mean=False,
                 backward_loss=True,
                 supervised=True,
                 sources_weight=0.5,
                 inactivity_threshold=-40.,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards
        """
        super().__init__()
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.supervised = supervised
        self.sources_weight=sources_weight
        self.permutations = list(itertools.permutations(
            torch.arange(2)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.inactivity_threshold = inactivity_threshold
        self.return_individual_results = return_individual_results

    def normalize_input(self, signals):
        if self.perform_zero_mean:
            return signals - torch.mean(signals, dim=-1, keepdim=True)
        return signals

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_snrs_and_inactive(self,
                                           pr_batch,
                                           t_batch,
                                           activity_mask,
                                           denom_stabilizer,
                                           eps=1e-9):
        # Compute SNR for the active and inactive sources
        nom = self.dot(t_batch, t_batch) + eps
        error = pr_batch - t_batch
        denom = self.dot(error, error) + denom_stabilizer + eps
        # return 10. * activity_mask * torch.log10(nom / denom + eps) - (
        #     (~activity_mask) * torch.abs(denom_stabilizer))
        # return - torch.abs(error)
        return 10. * activity_mask * torch.log10(nom / denom + eps)

    def compute_permuted_snrs(self,
                              permuted_pr_batch,
                              t_batch,
                              eps=1e-9):
        s_t = t_batch
        e_t = permuted_pr_batch - s_t
        snrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                (self.dot(e_t, e_t) + eps))
        return snrs

    def compute_snr(self,
                    pr_batch,
                    ref_sources_batch,
                    ref_noises_batch,
                    initial_mixture,
                    eps=1e-9,
                    thresh=0.001):

        mixture_power = self.dot(initial_mixture, initial_mixture)
        mixture_power_repeated = mixture_power.repeat(1, 2, 1)

        # Compute the maximum SNR obtained by the combinations of estimated
        # sources {0, 1} -> reference sources {0, 1} and estimated sources
        # estimated sources {2, 3} -> reference noises {0, 1}.
        if self.supervised:
            ref_sources_powers = self.dot(ref_sources_batch, ref_sources_batch)
            ref_noises_powers = self.dot(ref_noises_batch, ref_noises_batch)
            sources_input_snr = 10. * torch.log10(
                ref_sources_powers / (mixture_power + eps))
            noises_input_snr = 10. * torch.log10(
                ref_noises_powers / (mixture_power + eps))
            sources_activity_mask = sources_input_snr.ge(
                self.inactivity_threshold)
            noises_activity_mask = noises_input_snr.ge(
                self.inactivity_threshold)

            sources_active_stabilizer = sources_activity_mask * ref_sources_powers
            sources_inactive_stabilizer = (
                                              ~sources_activity_mask) * mixture_power_repeated
            sources_denom_stabilizer = thresh * (
                        sources_active_stabilizer + sources_inactive_stabilizer)

            noises_active_stabilizer = noises_activity_mask * ref_noises_powers
            noises_inactive_stabilizer = (
                                             ~noises_activity_mask) * mixture_power_repeated
            noises_denom_stabilizer = thresh * (
                        noises_active_stabilizer + noises_inactive_stabilizer)

            num_active_sources = sources_activity_mask.sum([-2, -1]).unsqueeze(-1)
            num_active_noises = noises_activity_mask.sum([-2, -1]).unsqueeze(-1)

            sources_snr_l = []
            for perm in self.permutations:
                permuted_pr_batch = pr_batch[:, :2][:, perm]
                snr = self.compute_permuted_snrs_and_inactive(
                    permuted_pr_batch, ref_sources_batch, sources_activity_mask,
                    sources_denom_stabilizer, eps=eps)
                sources_snr_l.append(snr)
            all_sources_snrs = torch.cat(sources_snr_l, -1)
            best_sources_snr, _ = torch.max(all_sources_snrs.sum(-2) * num_active_sources, -1)

            noises_snr_l = []
            for perm in self.permutations:
                permuted_pr_batch = pr_batch[:, 2:][:, perm]
                snr = self.compute_permuted_snrs_and_inactive(
                    permuted_pr_batch, ref_noises_batch, noises_activity_mask,
                    noises_denom_stabilizer, eps=eps)
                noises_snr_l.append(snr)
            all_noises_snrs = torch.cat(noises_snr_l, -1)
            best_noises_snr, _ = torch.max(
                all_noises_snrs.sum(-2) * num_active_noises, -1)

            best_snr = (self.sources_weight * best_sources_snr +
                        (1. - self.sources_weight) * best_noises_snr)
        else:
            # For the unsupervised case take the maximum SNR by reconstructing
            # the mixtures.
            ref_mixtures = ref_sources_batch + ref_noises_batch
            ref_mixtures_powers = self.dot(ref_mixtures, ref_mixtures)
            mixtures_input_snr = 10. * torch.log10(
                ref_mixtures_powers / (mixture_power + eps))
            mixtures_activity_mask = mixtures_input_snr.ge(self.inactivity_threshold)

            mixtures_active_stabilizer = mixtures_activity_mask * ref_mixtures_powers
            mixtures_inactive_stabilizer = (~mixtures_activity_mask) * mixture_power_repeated
            mixtures_denom_stabilizer = thresh * (
                    mixtures_active_stabilizer + mixtures_inactive_stabilizer)
            num_active_mixtures = mixtures_activity_mask.sum([-2, -1]).unsqueeze(-1)

            snr_l = []
            for perm in self.permutations:
                permuted_pr_batch = pr_batch[:, :2][:, perm]
                est_mixture_1 = permuted_pr_batch[:, 0:1] + pr_batch[:, 2:3]
                est_mixture_2 = permuted_pr_batch[:, 1:2] + pr_batch[:, 3:4]
                for ref_mix_perm in self.permutations:
                    permuted_ref_mixtures = ref_mixtures[:, ref_mix_perm]
                    permuted_mixtures_activity_mask = mixtures_activity_mask[:, ref_mix_perm]
                    permuted_mixtures_denom_stabilizer = mixtures_denom_stabilizer[:, ref_mix_perm]
                    snr_1 = self.compute_permuted_snrs_and_inactive(
                        est_mixture_1, permuted_ref_mixtures[:, 0:1],
                        permuted_mixtures_activity_mask[:, 0:1],
                        permuted_mixtures_denom_stabilizer[:, 0:1], eps=eps)
                    snr_2 = self.compute_permuted_snrs_and_inactive(
                        est_mixture_2, permuted_ref_mixtures[:, 1:2],
                        permuted_mixtures_activity_mask[:, 1:2],
                        permuted_mixtures_denom_stabilizer[:, 1:2], eps=eps)
                    # snr_reg = self.compute_permuted_snrs(
                    #     est_mixture_2, est_mixture_1)
                    # snr_l.append(snr_1 + snr_2 - snr_reg)
                    snr_l.append(snr_1 + snr_2)
            all_mix_snrs = torch.cat(snr_l, -1)
            best_snr, _ = torch.max(
                all_mix_snrs.sum(-2) * num_active_mixtures, -1)


        if not self.return_individual_results:
            best_snr = best_snr.mean()

        if self.backward_loss:
            return -best_snr
        return best_snr

    def forward(self,
                pr_batch,
                ref_sources_batch,
                ref_noises_batch,
                input_mixture,
                eps=1e-9):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param ref_sources_batch: Target wavs: Torch Tensors of size:
                        batch_size x 2 x length_of_wavs
        :param ref_noises_batch: Target wavs: Torch Tensors of size:
                        batch_size x 2 x length_of_wavs
        :param input_mixture: Target wavs: Torch Tensors of size:
                        batch_size x 1 x length_of_wavs
        :param eps: Numerical stability constant.
        """
        pr_batch = self.normalize_input(pr_batch)
        ref_sources_batch = self.normalize_input(ref_sources_batch)
        ref_noises_batch = self.normalize_input(ref_noises_batch)
        input_mixture = self.normalize_input(input_mixture)

        snr_l = self.compute_snr(
            pr_batch, ref_sources_batch, ref_noises_batch,
            input_mixture, eps=eps)

        return snr_l


class PermInvariantSNRwithZeroRefs(nn.Module):
    """!
    Class for SNR computation between reconstructed signals and
    target wavs with compensation for zero reference signals."""

    def __init__(self,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 valid_permutations=None,
                 inactivity_threshold=-40.,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards
        """
        super().__init__()
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        if valid_permutations is None:
            self.permutations = list(itertools.permutations(
                torch.arange(n_sources)))
        else:
            self.permutations = valid_permutations
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.n_sources = n_sources
        self.inactivity_threshold = inactivity_threshold
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

    def compute_permuted_snrs_and_inactive(self,
                                           pr_batch,
                                           t_batch,
                                           activity_mask,
                                           denom_stabilizer,
                                           eps=1e-9):
        # Compute SNR for the active and inactive sources
        nom = self.dot(t_batch, t_batch) + eps
        error = pr_batch - t_batch
        denom = self.dot(error, error) + denom_stabilizer + eps

        # return - 10. * torch.log10(denom + eps)
        # return 10. * activity_mask * torch.log10(nom) - 10. * torch.log(denom)
        return 10. * activity_mask * torch.log10(nom / denom + eps)

    def compute_permuted_snrs(self,
                              permuted_pr_batch,
                              t_batch,
                              eps=1e-9):
        s_t = t_batch
        e_t = permuted_pr_batch - s_t
        snrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                (self.dot(e_t, e_t) + eps))
        return snrs

    def compute_snr(self,
                    pr_batch,
                    t_batch,
                    eps=1e-9,
                    thresh=0.001):

        initial_mixture = torch.sum(t_batch, -2, keepdim=True)
        mixture_power = self.dot(initial_mixture, initial_mixture)
        mixture_power_repeated = mixture_power.repeat(1, self.n_sources, 1)
        target_powers = self.dot(t_batch, t_batch)
        input_snr = 10. * torch.log10(
            target_powers / (mixture_power + eps))
        activity_mask = input_snr.ge(self.inactivity_threshold)

        active_stabilizer = activity_mask * target_powers
        inactive_stabilizer = (~activity_mask) * mixture_power_repeated
        denom_stabilizer = thresh * (active_stabilizer + inactive_stabilizer)
        num_active_sources = activity_mask.sum([-2, -1]).unsqueeze(-1)

        snr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            snr = self.compute_permuted_snrs_and_inactive(
                permuted_pr_batch, t_batch, activity_mask, denom_stabilizer,
                eps=eps)
            snr_l.append(snr)
        all_snrs = torch.cat(snr_l, -1)
        # best_snr, best_perm_ind = torch.max(all_snrs.mean(-2), -1)
        best_snr, best_perm_ind = torch.max(
            all_snrs.sum(-2) * num_active_sources, -1)

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


class SimplerPermInvariantSNRwithZeroRefs(nn.Module):
    """!
    Class for SNR computation between reconstructed signals and
    target wavs with compensation for zero reference signals."""

    def __init__(self,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 inactivity_threshold=-40.,
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
        self.inactivity_threshold = inactivity_threshold
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

    def compute_permuted_snrs_and_inactive(self,
                                           pr_batch,
                                           t_batch,
                                           activity_mask,
                                           denom_stabilizer,
                                           eps=1e-9):
        # Compute SNR for the active and inactive sources
        nom = self.dot(t_batch, t_batch) + eps
        error = pr_batch - t_batch
        denom = self.dot(error, error) + denom_stabilizer + eps

        # return - 10. * torch.log10(denom)
        return 10. * activity_mask * torch.log10(nom / denom + eps)

    def compute_snr(self,
                    pr_batch,
                    t_batch,
                    eps=1e-9,
                    thresh=0.001):

        initial_mixture = torch.sum(t_batch, -2, keepdim=True)
        mixture_power = self.dot(initial_mixture, initial_mixture)
        mixture_power_repeated = mixture_power.repeat(1, self.n_sources, 1)
        target_powers = self.dot(t_batch, t_batch)
        input_snr = 10. * torch.log10(
            target_powers / (mixture_power + eps))
        activity_mask = input_snr.ge(self.inactivity_threshold)

        active_stabilizer = activity_mask * target_powers
        inactive_stabilizer = (~activity_mask) * mixture_power_repeated
        denom_stabilizer = thresh * (active_stabilizer + inactive_stabilizer)
        num_active_sources = activity_mask.sum([-2, -1]).unsqueeze(-1)

        snr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            snr = self.compute_permuted_snrs_and_inactive(
                permuted_pr_batch, t_batch, activity_mask, denom_stabilizer,
                eps=eps)
            snr_l.append(snr)
        all_snrs = torch.cat(snr_l, -1)
        best_snr, best_perm_ind = torch.max(
            all_snrs.sum(-2) * num_active_sources, -1)

        if not self.return_individual_results:
            best_snr = best_snr.mean()

        if self.backward_loss:
            return -best_snr
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
    improvement = False

    snr_loss = PermInvariantSNRwithZeroRefs(n_sources=n_sources,
                                            zero_mean=zero_mean,
                                            backward_loss=True,
                                            improvement=improvement)

    pr_batch = torch.zeros((bs, n_sources, n_samples), dtype=torch.float32)
    t_batch = torch.zeros((bs, n_sources, n_samples), dtype=torch.float32)

    pr_batch[:, 0:2, :] = 2.
    t_batch[:, 1:-1, :] = 2.

    snr_result, best_permutation = snr_loss(pr_batch, t_batch,
                                            eps=1e-9,
                                            return_best_permutation=True)
    print(snr_result, best_permutation)

if __name__ == "__main__":
    test_snr_implementation()
