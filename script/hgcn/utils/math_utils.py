"""Math utils functions."""

import torch


def cosh(x, clamp=10):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=10):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=10):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        a = 1 + z.pow(2)
        assert torch.min(a) > 0
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-10).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        a = 1 + input ** 2
        assert torch.min(a) > 0
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-5)
        ctx.save_for_backward(x)
        z = x.double()
        a = z.pow(2) - 1
        assert torch.min(a) > 0
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-10).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        a = input ** 2 - 1
        assert torch.min(a) > 0
        return grad_output / (input ** 2 - 1) ** 0.5


def darcosh(x):
    cond = (x < 1 + 1e-5)
    x = torch.where(cond, 2 * torch.ones_like(x), x)
    a = x**2 - 1
    assert torch.min(a) > 0
    x = torch.where(~cond, 2 * arcosh(x) / torch.sqrt(x**2 - 1), x)
    return x


def d2arcosh(x):
    cond = (x < 1 + 1e-5)
    x = torch.where(cond, -2/3 * torch.ones_like(x), x)
    a = x**2 - 1
    assert torch.min(a) > 0
    x = torch.where(~cond, 2 / (x**2 - 1) - 2 * x * arcosh(x) / ((x**2 - 1)**(3/2)), x)
    return x