import torch

from pointwise_methods import get_argmax, get_quantile, get_mean, get_mean_std


def get_pairwise_preds(probs):
    # function called by get_pairwise_ranking_preds and get_pairwise_scoring_preds
    ## probs is probability distribution over delta scores or preferences (dim=-1)
    ## with the two presentation orders interleaved (dim=-2).
    ## assume ties have been handled, since they are handled differently for scoring vs ranking.

    k = probs.shape[-1]
    assert k % 2 == 1, k
    n_instance = probs.shape[-2]
    assert n_instance % 2 == 0, n_instance
    support = torch.linspace(k // 2, -(k // 2), k)

    sym_probs = (probs[..., ::2, :] + torch.flip(probs[..., 1::2, :], [-1])) / 2

    # s1 and s2 denote pre- and post-aggregation, respectively
    method_names = ["mode_s2", "mode_s1", "medi_s2", "medi_s1", "mean_s2", "mean_s1"]
    method_preds = torch.zeros(len(method_names), *probs.shape[:-2], n_instance // 2)

    # mode_s1

    mode_s1s = torch.sign(get_argmax(sym_probs, support, split_ties=True))
    method_preds[method_names.index("mode_s1")] = mode_s1s

    # mode_s2

    modes = get_argmax(probs, support, split_ties=True)
    # symmetrize and normalize
    mode_s2_numers = modes[..., ::2] - modes[..., 1::2]
    mode_s2_denoms = torch.abs(modes[..., ::2]) + torch.abs(modes[..., 1::2])
    mode_s2s = torch.where(
        mode_s2_denoms > 0, mode_s2_numers / mode_s2_denoms, torch.zeros_like(mode_s2_denoms)
    )
    method_preds[method_names.index("mode_s2")] = mode_s2s

    # medi_s1

    medi_s1s = torch.sign(get_quantile(sym_probs, 0.5, support, split_ties=True))
    method_preds[method_names.index("medi_s1")] = medi_s1s

    # medi_s2

    medis = get_quantile(probs, 0.5, support, split_ties=True)
    medi_s2_numers = medis[..., ::2] - medis[..., 1::2]
    medi_s2_denoms = torch.abs(medis[..., ::2]) + torch.abs(medis[..., 1::2])
    medi_s2s = torch.where(
        medi_s2_denoms > 0, medi_s2_numers / medi_s2_denoms, torch.zeros_like(medi_s2_denoms)
    )
    method_preds[method_names.index("medi_s2")] = medi_s2s

    # mean_s1

    means, stds = get_mean_std(sym_probs, support=support)
    mean_abss = get_mean(sym_probs, support=support.abs())
    mean_denoms = mean_abss + stds
    mean_s1s = torch.where(mean_denoms > 1e-4, means / mean_denoms, torch.zeros_like(mean_denoms))
    method_preds[method_names.index("mean_s1")] = mean_s1s

    # mean_s2

    means, stds = get_mean_std(probs, support=support)
    # normalize unidirectional means
    mean_abss = get_mean(probs, support=support.abs())
    mean_denoms = mean_abss + stds
    means = torch.where(mean_denoms > 1e-4, means / mean_denoms, torch.zeros_like(mean_denoms))
    # symmetrize
    mean_s2s = (means[..., ::2] - means[..., 1::2]) / 2
    method_preds[method_names.index("mean_s2")] = mean_s2s

    # check that predictions are in [-1, 1]
    assert torch.allclose(method_preds, torch.clamp(method_preds, -1, 1), atol=1e-4), (
        method_preds.min(dim=-1),
        method_preds.max(dim=-1),
    )
    method_preds = torch.clamp(method_preds, -1, 1)

    return method_names, method_preds


def get_pairwise_ranking_preds(probs):
    # pairwise ranking
    ## probs is probability distribution over k preference options (k//2, ..., -(k//2)) (dim=-1)
    ## with the two presentation orders interleaved (dim=-2)

    k = probs.shape[1]

    if k == 2:  # convert to L3: P(=) = 0
        probs = torch.cat(
            [probs[..., :1], torch.zeros_like(probs[..., :1]), probs[..., 1:]], dim=-1
        )
        k = 3  # change k
        # set near-ties to one-hot on tie
        probs[torch.isclose(probs[..., 0], probs[..., 2], atol=1e-6)] = torch.nn.functional.one_hot(
            torch.tensor(1), num_classes=k
        ).float()

    # set nans to tie
    probs[torch.any(torch.isnan(probs), dim=-1)] = torch.nn.functional.one_hot(
        torch.tensor(k // 2), num_classes=k
    ).float()

    return get_pairwise_preds(probs)


def get_pairwise_scoring_preds(probs):
    # pairwise scoring
    ## probs is probability distribution over k scores (dim=-1)
    ## with the two presentation orders interleaved (dim=-3).
    ## in dim=-2, i=0 and i=1 are for the responses presented 1st and 2nd, respectively.

    # if nan, set both to lowest score
    k = probs.shape[-1]
    probs[torch.any(torch.isnan(probs), dim=(-1, -2))] = torch.nn.functional.one_hot(
        torch.tensor(0), num_classes=k
    ).float()

    # compute dr_probs: probability distribution over delta score X2 - X1.
    # delta score is in [-(k-1), k-1]. we shift up by k-1 to get idxs.
    dr_probs = torch.zeros(*probs.shape[:-2], k * 2 - 1)
    # treat X1 as a constant equal to the greedily decoded score
    argmax_1s = torch.argmax(probs[..., 0, :], dim=-1)
    score_idxs = torch.arange(k) - argmax_1s[..., None] + (k - 1)
    dr_probs.scatter_(1, score_idxs, probs[..., 1, :])

    return get_pairwise_preds(dr_probs)
