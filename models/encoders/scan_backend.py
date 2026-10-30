"""Selective scan fallback with ordinary PyTorch autograd.

The CUDA extension remains the preferred training backend. This reference
recurrence also permits correctness checks on machines without that extension.
"""
import warnings

import torch
import torch.nn.functional as F


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=True, nrows=1):
    dtype = u.dtype
    u, delta, A, B, C = [value.float() for value in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    if B.ndim == 3:
        B = B.unsqueeze(1)
    if C.ndim == 3:
        C = C.unsqueeze(1)
    batch, channels, length = u.shape
    groups = B.shape[1]
    if channels % groups:
        raise ValueError('Scan channels must be divisible by B/C groups')
    B = B.repeat_interleave(channels // groups, dim=1)
    C = C.repeat_interleave(channels // C.shape[1], dim=1)
    state = u.new_zeros(batch, channels, A.shape[-1])
    output = []
    for index in range(length):
        dt = delta[:, :, index, None]
        state = (torch.exp(dt * A[None]) * state
                 + dt * B[:, :, :, index] * u[:, :, index, None])
        output.append((state * C[:, :, :, index]).sum(-1))
    result = torch.stack(output, dim=-1)
    if D is not None:
        result = result + u * D.float()[None, :, None]
    return result.to(dtype)


_warned = False


def warn_reference_backend():
    global _warned
    if not _warned:
        warnings.warn('Using the slower PyTorch reference scan. '
                      'Build models/encoders/selective_scan '
                      'for full-size CUDA training.', RuntimeWarning, stacklevel=2)
        _warned = True
