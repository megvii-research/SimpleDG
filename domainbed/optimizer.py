import torch


class SAM(torch.optim.Optimizer):
    """An implementation of paper `SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY IMPROVING GENERALIZATION`
    code borrowed from https://github.com/davda54/sam
    """
    def __init__(self, params, base_optimizer, rho=0.03, adaptive=False, defaults=None, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs) if defaults is None else defaults
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

        return grad_norm

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # get back to "w" from "w + e(w)"
                self.state[p]['e_w'] = 0

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        grad_norm = self.first_step(zero_grad=True)
        closure()
        self.second_step()

        return grad_norm

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class GSAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.03, alpha=0.1, adaptive=False, **kwargs):
        defaults = dict(rho=rho, alpha=alpha, adaptive=adaptive, **kwargs)
        super().__init__(params, base_optimizer, rho, adaptive, defaults=defaults, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                self.state[p]['adv_grad'] = p.grad.clone()

        if zero_grad: self.zero_grad()

        return grad_norm

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            alpha = group['alpha']
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # get back to "w" from "w + e(w)"
                p_vertical = self.decompose(self.state[p]['adv_grad'], p.grad)
                p.grad.sub_(alpha * p_vertical)

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def decompose(self, adv_grad, grad):
        grad_parallel = (adv_grad * grad).sum() / (grad**2).sum() * grad
        grad_vertical = adv_grad - grad_parallel
        return grad_vertical

class ESAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, beta=0.6, gamma=0.5, adaptive=False, **kwargs):
        defaults = dict(rho=rho, beta=beta, adaptive=adaptive, **kwargs)
        self.gamma = gamma
        super().__init__(params, base_optimizer, rho, adaptive, defaults, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) / group['beta']

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.requires_grad = True

        if zero_grad: self.zero_grad()

        return grad_norm

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]['e_w'])  # get back to "w" from "w + e(w)"
                self.state[p]['e_w'] = 0

                if torch.rand(1)[0] > group['beta']:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # closure = torch.enable_grad()(closure)

        loss_before = closure()

        loss = loss_before.mean()
        loss.backward()

        grad_norm = self.first_step(zero_grad=True)

        loss_after = closure()

        instance_sharpness = loss_after - loss_before
        cutoff, _ = torch.topk(instance_sharpness, int(loss_after.shape[0] * self.gamma))
        cutoff = cutoff[-1]
        indices = [instance_sharpness > cutoff]
        loss = loss_after[indices].mean()
        loss.backward()

        self.second_step(zero_grad=True)

        return loss_before.mean(), grad_norm
