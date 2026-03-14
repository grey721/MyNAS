"""
ea.proxy — Zero-Cost Proxy public API.

Overview
--------
A *zero-cost proxy* (ZCP) estimates a network's final accuracy without any
gradient-based training.  Each proxy is a plain Python class that maps a
randomly-initialised ``nn.Module`` + a single data batch to a scalar score
(higher is better).

This package provides:

* ``BaseProxy`` — abstract base class; subclass to implement a new proxy.
* ``CompositeProxy`` — linearly combines multiple proxies by weighted sum.
* Two built-in proxies: ``NasWotProxy`` and ``SynFlowProxy``.

Architecture
------------
::

    ea/proxy/
    ├── __init__.py   ← this file; re-exports the public API
    ├── base.py       ← BaseProxy ABC + CompositeProxy
    ├── naswot.py     ← NasWotProxy
    └── synflow.py    ← SynFlowProxy

There is no registry or auto-registration.  Import whichever class you need
and instantiate it directly.

Quick start
-----------
**Single proxy**::

    from ea.proxy.naswot import NasWotProxy

    proxy = NasWotProxy(batch_size=128)
    score = proxy.score(net, batch)   # float, higher is better

**Composite proxy (weighted sum)**::

    from ea.proxy import CompositeProxy
    from ea.proxy.naswot  import NasWotProxy
    from ea.proxy.synflow import SynFlowProxy

    proxy = CompositeProxy(
        proxies=[NasWotProxy(batch_size=128), SynFlowProxy()],
        weights=[0.7, 0.3],   # auto-normalised; equal weights if omitted
    )
    score = proxy.score(net, batch)

Built-in proxies
----------------
+--------------------+------------------+-----------------------------------------------+
| Class              | Module           | Score signal                                  |
+====================+==================+===============================================+
| ``NasWotProxy``    | naswot.py        | log|K|: log-det of the ReLU activation kernel |
+--------------------+------------------+-----------------------------------------------+
| ``SynFlowProxy``   | synflow.py       | sum(|θ ⊙ ∂R/∂θ|): data-free synaptic flow   |
+--------------------+------------------+-----------------------------------------------+

Adding a new proxy
------------------
1. Create ``ea/proxy/<your_proxy>.py``.
2. Subclass ``BaseProxy`` and implement ``_compute(net, batch) -> float``.
   Both ``net`` and ``batch`` are already on GPU inside ``_compute``; do not
   call ``.cuda()`` again.  Return value must be higher-is-better.
3. Import and instantiate the class wherever it is needed.

No other files need to change.  Example::

    # ea/proxy/grad_norm.py
    import torch.nn as nn
    from ea.proxy.base import BaseProxy

    class GradNormProxy(BaseProxy):
        \"\"\"Gradient-norm proxy: sum of squared parameter gradients.\"\"\"

        def _compute(
            self,
            net: nn.Module,
            batch: tuple[torch.Tensor, torch.Tensor],
        ) -> float:
            x, target = batch
            net.zero_grad()
            loss = nn.CrossEntropyLoss()(net(x), target)
            loss.backward()
            score = sum(
                p.grad.norm().item() ** 2
                for p in net.parameters()
                if p.grad is not None
            )
            net.zero_grad()
            return score   # higher is better

Usage::

    from ea.proxy.grad_norm import GradNormProxy
    from ea.evaluate import Evaluator

    evaluator = Evaluator(
        'example_arch', 'cifar10',
        batch_size=128,
        proxy=GradNormProxy(),
    )

Score convention
----------------
All proxies must return a **higher-is-better** float from ``_compute()``.
If your metric is naturally lower-is-better (e.g. a loss), negate it::

    return -loss_value

The ``Evaluator`` stores the negated score as ``err`` (lower-is-better) in
column ``COL_ERR`` of the fitness matrix; do not negate twice.
"""

from ea.proxy.base    import BaseProxy, CompositeProxy
from ea.proxy.naswot  import NASWOT
from ea.proxy.synflow import SynFlow

__all__ = [
    'BaseProxy',
    'CompositeProxy',
    'NASWOT',
    'SynFlow',
]
