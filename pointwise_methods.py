import torch

# p1 and p2 are tensors of equal shape that are pairs of probability distributions over 1 to k (k = p1.shape[-1]).
# functions starting with 'diff' return a tensor shaped p1.shape[:-1] with values in the interval [-1, 1],
# where a higher value means p1 is better and a lower value means p2 is better.


def get_argmax(p, support=None, split_ties=False):
    if support is None:
        support = torch.linspace(0, 1, p.shape[-1])

    if not split_ties:
        return support[torch.argmax(p, dim=-1)]
    else:
        maxs = torch.max(p, dim=-1).values
        max_masks = p == maxs[..., None]
        return torch.sum(support * max_masks, dim=-1) / torch.sum(max_masks, dim=-1)


def diff_argmax(p1, p2, split_ties=True):
    return torch.sign(get_argmax(p1, split_ties=split_ties) - get_argmax(p2, split_ties=split_ties))


def get_mean(p, support=None):
    if support is None:
        support = torch.linspace(0, 1, p.shape[-1])
    return (support * p).sum(dim=-1)


def get_round_mean(p):
    k = p.shape[-1]
    return torch.round(get_mean(p) * (k - 1)) / (k - 1)


def diff_round_mean(p1, p2):
    return torch.sign(get_round_mean(p1) - get_round_mean(p2))


def get_quantile(p, q, support=None, split_ties=False):
    if support is None:
        support = torch.linspace(0, 1, p.shape[-1])

    # argmax takes the first occurrence of the maximum
    if not split_ties:
        return support[torch.argmax((torch.cumsum(p, dim=-1) > q).long(), dim=-1)]
    else:
        cumsum = torch.cumsum(p, dim=-1)
        return (
            support[torch.argmax((cumsum > q).long(), dim=-1)]
            + support[torch.argmax((cumsum >= q).long(), dim=-1)]
        ) / 2


def get_median(p):
    return get_quantile(p, 0.5)


def diff_median(p1, p2):
    return torch.sign(get_median(p1) - get_median(p2))


def get_lower_percentile(p):
    return get_quantile(p, 0.01)


def diff_lower_percentile(p1, p2):
    return torch.sign(get_lower_percentile(p1) - get_lower_percentile(p2))


def diff_quantiles(p1, p2):
    # compute cdfs (endpoints are redundant)
    c1 = torch.cumsum(p1[..., :-1], dim=-1)
    c2 = torch.cumsum(p2[..., :-1], dim=-1)
    # merge into shape (..., 2, 2(k-1)), where row 0 is where the quantile increments
    # and row 1 indicates either p1 or p2
    merged = torch.stack(
        (
            torch.cat([c1, c2], dim=-1),
            torch.cat([torch.ones_like(c1), -torch.ones_like(c2)], dim=-1),
        ),
        dim=-2,
    )
    # sort quantiles (sort cols by row 0)
    quantile_sort = torch.argsort(merged[..., 0:1, :], dim=-1).expand_as(merged)
    merged = torch.gather(merged, -1, quantile_sort)
    # compute who is in the lead at each quantile (endpoints are redundant)
    signs = torch.sign(torch.cumsum(merged[..., 1, :-1], dim=-1))
    # compute increments
    incs = merged[..., 0, 1:] - merged[..., 0, :-1]
    # return
    return (signs * incs).sum(dim=-1)


def diff_probsup(p1, p2):
    p_1_le_2 = (torch.cumsum(p1, dim=-1) * p2).sum(dim=-1)
    p_1_ge_2 = 1 - p_1_le_2 + (p1 * p2).sum(dim=-1)
    return p_1_ge_2 - p_1_le_2


def diff_mean(p1, p2):
    joint_p = p1.unsqueeze(-1) * p2.unsqueeze(-2)
    support = torch.linspace(0, 1, p1.shape[-1])
    # support_2d[i, j] = support[i] - support[j]
    support_2d = support.unsqueeze(-1) - support.unsqueeze(-2)
    torqs = joint_p * support_2d
    mean_diff = torqs.sum(dim=(-1, -2))  # E(X1 - X2)
    mean_abs_diff = torqs.abs().sum(dim=(-1, -2))  # E|X1 - X2|

    # compute std(X1 - X2)
    sq_torqs = joint_p * (support_2d - mean_diff[..., None, None]) ** 2
    std_diff = torch.sqrt(sq_torqs.sum(dim=(-1, -2)))

    denom = mean_abs_diff + std_diff
    return torch.where(denom > 1e-3, mean_diff / denom, torch.zeros_like(denom))


def get_mean_stdd(p):
    # get mean and lower semi-deviation

    mean = get_mean(p)
    support = torch.linspace(0, 1, p.shape[-1])
    # sqdevs: max(0, mean - x)^2. expand mean and support to p.shape.
    sqdevs = (
        torch.max(torch.zeros(()), mean[..., None] - support.view([1] * mean.dim() + [-1])) ** 2
    )
    stdd = torch.sqrt((p * sqdevs).sum(dim=-1))
    return mean, stdd


def diff_ram(p1, p2):
    joint_p = p1.unsqueeze(-1) * p2.unsqueeze(-2)

    mean1, stdd1 = get_mean_stdd(p1)
    mean2, stdd2 = get_mean_stdd(p2)

    # compute E|(X1 - stdd1) - (X2 - stdd2)|
    support = torch.linspace(0, 1, p1.shape[-1])
    support_2d = support.unsqueeze(-1) - support.unsqueeze(-2)
    support_2d_with_stdd = support_2d - (stdd1 - stdd2)[..., None, None]
    torqs = joint_p * support_2d_with_stdd
    mean_abs_diff_with_stdd = torqs.abs().sum(dim=(-1, -2))

    # compute std((X1 - stdd1) - (X2 - stdd2)) = std(X1 - X2)
    mean_diff = mean1 - mean2
    sq_torqs = joint_p * (support_2d - mean_diff[..., None, None]) ** 2
    std_diff = torch.sqrt(sq_torqs.sum(dim=(-1, -2)))

    denom = mean_abs_diff_with_stdd + std_diff
    return torch.where(
        denom > 1e-3, ((mean1 - stdd1) - (mean2 - stdd2)) / denom, torch.zeros_like(denom)
    )


def get_mean_std(p, support=None):
    if support is None:
        support = torch.linspace(0, 1, p.shape[-1])
    mean = get_mean(p, support)
    # sqdevs: (mean - x)^2. expand mean and support to p.shape.
    sqdevs = (mean[..., None] - support.view([1] * mean.dim() + [-1])) ** 2
    std = torch.sqrt((p * sqdevs).sum(dim=-1))
    return mean, std
