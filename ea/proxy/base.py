"""
Zero-cost proxy base class and composite wrapper.

Design
------
Proxies are plain Python classes — no registry, no decorators, no magic
imports.  To use a proxy, import its class directly and instantiate it::

    from ea.proxy.naswot  import NasWotProxy
    from ea.proxy.synflow import SynFlowProxy

    proxy = NasWotProxy(batch_size=128)
    score = proxy.score(net, batch)   # float, higher is better

To combine multiple proxies, use ``CompositeProxy``::

    from ea.proxy import CompositeProxy

    proxy = CompositeProxy(
        proxies=[NasWotProxy(batch_size=128), SynFlowProxy()],
        weights=[0.7, 0.3],
    )

Adding a new proxy
------------------
1. Create ``ea/proxy/<your_proxy>.py``.
2. Subclass ``BaseProxy`` and implement ``_compute(net, batch) -> float``.
   *net* and *batch* are already on GPU inside ``_compute``; no ``.cuda()``
   needed.  The return value must be higher-is-better.
3. Import and instantiate the new class wherever it is needed.

No other files need to change.

Score convention
----------------
``score()`` returns higher-is-better values ("fitness" direction).
If your metric is naturally lower-is-better (e.g. a loss), negate it::

    return -loss_value
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseProxy(abc.ABC):
    """
    Abstract base class for zero-cost proxies.

    Subclasses only need to implement ``_compute(net, batch) -> float``,
    where the return value is higher-is-better.
    """

    def score(
        self,
        net: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Compute a scalar proxy score for one network (higher is better).

        Parameters
        ----------
        net:
            Network to evaluate; expected on CPU on entry.
            This method moves it to GPU internally.
        batch:
            ``(x, target)`` from a DataLoader, on CPU.

        Returns
        -------
        float
            Higher values indicate a better network.
        """
        net = net.cuda()
        x, target = batch
        x      = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        try:
            result = self._compute(net, (x, target))
        finally:
            del x, target

            # net.cpu()
            # torch.cuda.empty_cache()

        return float(result)

    @abc.abstractmethod
    def _compute(
        self,
        net: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Subclass hook: run the scoring computation on GPU.

        Both *net* and *batch* are already on GPU; no ``.cuda()`` needed.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Composite proxy (weighted sum)
# ---------------------------------------------------------------------------

class CompositeProxy(BaseProxy):
    """
    Combine multiple proxies into a single fitness via weighted summation.

    Parameters
    ----------
    proxies:
        List of proxy instances.
    weights:
        Corresponding weight list (default: equal weights).
        Automatically normalised to sum to 1.

    Example
    -------
    ::

        from ea.proxy import CompositeProxy
        from ea.proxy.naswot  import NasWotProxy
        from ea.proxy.synflow import SynFlowProxy

        proxy = CompositeProxy(
            proxies=[NasWotProxy(batch_size=128), SynFlowProxy()],
            weights=[0.7, 0.3],
        )
        score = proxy.score(net, batch)
    """

    def __init__(
        self,
        proxies: list[BaseProxy],
        weights: list[float] | None = None,
    ) -> None:
        if not proxies:
            raise ValueError("proxies list must not be empty.")
        if weights is None:
            weights = [1.0] * len(proxies)
        if len(weights) != len(proxies):
            raise ValueError(
                f"Length mismatch: {len(proxies)} proxies but {len(weights)} weights."
            )

        total = sum(weights)
        self.proxies = proxies
        self.weights = [w / total for w in weights]

    def score(
        self,
        net: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """Invoke each sub-proxy in sequence and return the weighted sum."""
        total = 0.0
        for proxy, w in zip(self.proxies, self.weights):
            total += w * proxy.score(net, batch)
        return total

    def _compute(self, net, batch) -> float:
        # CompositeProxy overrides score(); this method is never called.
        raise NotImplementedError

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{p!r}x{w:.3f}" for p, w in zip(self.proxies, self.weights)
        )
        return f"CompositeProxy([{parts}])"


__all__ = ['BaseProxy', 'CompositeProxy']
