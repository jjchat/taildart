from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Distribution, constraints
from torch.distributions.utils import broadcast_all

from gluonts.torch.distributions import DistributionOutput, StudentTOutput


def _finfo_eps(tensor: Tensor) -> float:
    return torch.finfo(tensor.dtype).eps


def _squeeze_last(*args: Tensor) -> Tuple[Tensor, ...]:
    return tuple(arg.squeeze(-1) for arg in args)


class HansenSkewT(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "skew": constraints.interval(-1.0, 1.0),
        "df": constraints.greater_than(2.0),
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        skew: Tensor | float,
        df: Tensor | float,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.loc, self.scale, self.skew, self.df = broadcast_all(loc, scale, skew, df)
        self._c, self._a, self._b = self._compute_shape_constants(
            self.skew, self.df
        )
        self._standard_t = torch.distributions.StudentT(
            self.df,
            torch.zeros_like(self.loc),
            torch.sqrt((self.df - 2.0) / self.df),
        )
        super().__init__(self.loc.size(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HansenSkewT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.skew = self.skew.expand(batch_shape)
        new.df = self.df.expand(batch_shape)
        new._c = self._c.expand(batch_shape)
        new._a = self._a.expand(batch_shape)
        new._b = self._b.expand(batch_shape)
        new._standard_t = self._standard_t.expand(batch_shape)
        super(HansenSkewT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.loc

    @staticmethod
    def _compute_shape_constants(skew: Tensor, df: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        c = torch.exp(
            torch.lgamma(0.5 * (df + 1.0))
            - torch.lgamma(0.5 * df)
            - 0.5 * torch.log(torch.full_like(df, math.pi) * (df - 2.0))
        )
        a = 4.0 * skew * c * ((df - 2.0) / (df - 1.0))
        b = torch.sqrt((1.0 + 3.0 * skew.square() - a.square()).clamp_min(_finfo_eps(df)))
        return c, a, b

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        loc = self.loc.expand(shape)
        scale = self.scale.expand(shape)
        skew = self.skew.expand(shape)
        a = self._a.expand(shape)
        b = self._b.expand(shape)

        base = self._standard_t.sample(sample_shape).abs()
        prob_pos = ((1.0 + skew) * 0.5).clamp(0.0, 1.0)
        is_pos = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) < prob_pos
        signed = torch.where(is_pos, base, -base)
        factor = torch.where(is_pos, 1.0 + skew, 1.0 - skew)
        standardized = (factor * signed - a) / b
        return loc + scale * standardized

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        z = (value - self.loc) / self.scale
        threshold = -self._a / self._b
        denom = torch.where(z < threshold, 1.0 - self.skew, 1.0 + self.skew)
        scaled = (self._b * z + self._a) / denom
        return (
            torch.log(self._b)
            + torch.log(self._c)
            - torch.log(self.scale)
            - 0.5 * (self.df + 1.0) * torch.log1p(scaled.square() / (self.df - 2.0))
        )


class NormalInverseGaussian(Distribution):
    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.real,
        "loc": constraints.real,
        "delta": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        alpha: Tensor | float,
        beta: Tensor | float,
        loc: Tensor | float,
        delta: Tensor | float,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.alpha, self.beta, self.loc, self.delta = broadcast_all(
            alpha, beta, loc, delta
        )
        self._gamma = torch.sqrt((self.alpha.square() - self.beta.square()).clamp_min(0.0))
        super().__init__(self.alpha.size(), validate_args=validate_args)
        if self._validate_args and not (self.beta.abs() < self.alpha).all():
            raise ValueError("NormalInverseGaussian requires |beta| < alpha")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NormalInverseGaussian, _instance)
        batch_shape = torch.Size(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.beta = self.beta.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.delta = self.delta.expand(batch_shape)
        new._gamma = self._gamma.expand(batch_shape)
        super(NormalInverseGaussian, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.loc + self.delta * self.beta / self._gamma.clamp_min(
            _finfo_eps(self._gamma)
        )

    def _sample_inverse_gaussian(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tensor:
        shape = self._extended_shape(sample_shape)
        gamma = self._gamma.expand(shape).clamp_min(_finfo_eps(self._gamma))
        delta = self.delta.expand(shape)
        mu = delta / gamma
        lam = delta.square()

        normal = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        y = normal.square()
        sqrt_term = torch.sqrt(4.0 * mu * lam * y + mu.square() * y.square())
        x = mu + (mu.square() * y) / (2.0 * lam) - (mu / (2.0 * lam)) * sqrt_term

        accept_prob = mu / (mu + x)
        uniform = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return torch.where(uniform <= accept_prob, x, mu.square() / x.clamp_min(_finfo_eps(x)))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        beta = self.beta.expand(shape)
        loc = self.loc.expand(shape)
        v = self._sample_inverse_gaussian(sample_shape)
        z = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return loc + beta * v + torch.sqrt(v) * z

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        centered = value - self.loc
        q = torch.sqrt(self.delta.square() + centered.square())
        alpha_q = (self.alpha * q).clamp_min(_finfo_eps(q))
        # requires torch >= 2.0
        # scaled_modified_bessel_k1 is unsupported on MPS; compute on CPU and move back.
        if alpha_q.device.type == "mps":
            scaled_k1 = torch.special.scaled_modified_bessel_k1(alpha_q.cpu()).to(alpha_q.device)
        else:
            scaled_k1 = torch.special.scaled_modified_bessel_k1(alpha_q)
        log_k1 = torch.log(scaled_k1.clamp_min(_finfo_eps(scaled_k1))) - alpha_q

        return (
            torch.log(self.alpha)
            + torch.log(self.delta)
            - math.log(math.pi)
            - 0.5 * torch.log(self.delta.square() + centered.square())
            + log_k1
            + self.delta * self._gamma
            + self.beta * centered
        )


class JohnsonSU(Distribution):
    arg_constraints = {
        "gamma": constraints.real,
        "delta": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        gamma: Tensor | float,
        delta: Tensor | float,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.gamma, self.delta, self.loc, self.scale = broadcast_all(
            gamma, delta, loc, scale
        )
        super().__init__(self.gamma.size(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(JohnsonSU, _instance)
        batch_shape = torch.Size(batch_shape)
        new.gamma = self.gamma.expand(batch_shape)
        new.delta = self.delta.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(JohnsonSU, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        inv_delta = self.delta.reciprocal()
        return self.loc - self.scale * torch.exp(
            0.5 * inv_delta.square()
        ) * torch.sinh(self.gamma * inv_delta)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        normal = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        gamma = self.gamma.expand(shape)
        delta = self.delta.expand(shape)
        loc = self.loc.expand(shape)
        scale = self.scale.expand(shape)
        return loc + scale * torch.sinh((normal - gamma) / delta)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        scale = self.scale.clamp_min(_finfo_eps(self.scale))
        centered = (value - self.loc) / scale
        asinh_centered = torch.asinh(centered)
        z = self.gamma + self.delta * asinh_centered
        return (
            torch.log(self.delta)
            - torch.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * z.square()
            - 0.5 * torch.log1p(centered.square())
        )


class HansenSkewTOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1, "skew": 1, "df": 1}
    distr_cls: type = HansenSkewT

    @classmethod
    def domain_map(
        cls,
        loc: Tensor,
        scale: Tensor,
        skew: Tensor,
        df: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale_eps = _finfo_eps(scale)
        scale = F.softplus(scale).clamp_min(scale_eps)
        skew = torch.tanh(skew)
        # Keep df away from 2.0 so the standardized Student-t helper used by
        # HansenSkewT stays numerically valid under training noise.
        df = F.softplus(df) + 2.05
        return _squeeze_last(loc, scale, skew, df)

    @property
    def event_shape(self) -> Tuple:
        return ()


class NormalInverseGaussianOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"alpha": 1, "beta": 1, "loc": 1, "delta": 1}
    distr_cls: type = NormalInverseGaussian

    @classmethod
    def domain_map(
        cls,
        alpha: Tensor,
        beta: Tensor,
        loc: Tensor,
        delta: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        eps = _finfo_eps(alpha)
        alpha = F.softplus(alpha).clamp_min(eps)
        # Keep beta strictly inside (-alpha, alpha); tanh can saturate to +/-1.0
        # numerically under large activations, which violates the NIG constraint.
        beta = alpha * (1.0 - 4.0 * eps) * torch.tanh(beta)
        delta = F.softplus(delta).clamp_min(eps)
        return _squeeze_last(alpha, beta, loc, delta)

    @property
    def event_shape(self) -> Tuple:
        return ()


class JohnsonSUOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"gamma": 1, "delta": 1, "loc": 1, "scale": 1}
    distr_cls: type = JohnsonSU

    @classmethod
    def domain_map(
        cls,
        gamma: Tensor,
        delta: Tensor,
        loc: Tensor,
        scale: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        eps = _finfo_eps(delta)
        delta = F.softplus(delta).clamp_min(eps) + 0.5
        scale = F.softplus(scale).clamp_min(eps)
        return _squeeze_last(gamma, delta, loc, scale)

    @property
    def event_shape(self) -> Tuple:
        return ()


class Mixture(Distribution):
    """Finite mixture over heterogeneous scalar component distributions.

    The gate is expressed as unnormalized ``mixture_logits`` so the exact
    mixture NLL is ``-logsumexp(log_softmax(logits) + component_log_prob)``,
    which is numerically stable and keeps every component (and the gate)
    differentiable. There is intentionally no hard top-1 routing: output-head
    uncertainty over components must be retained.
    """

    arg_constraints: Dict[str, constraints.Constraint] = {}
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        mixture_logits: Tensor,
        components: Sequence[Distribution],
        validate_args: Optional[bool] = None,
    ) -> None:
        if len(components) < 2:
            raise ValueError("Mixture requires at least two component distributions.")
        self.mixture_logits = mixture_logits
        self.components = list(components)
        if int(mixture_logits.shape[-1]) != len(self.components):
            raise ValueError(
                f"mixture_logits last dimension ({int(mixture_logits.shape[-1])}) must "
                f"equal the number of components ({len(self.components)})."
            )
        batch_shape = mixture_logits.shape[:-1]
        for index, component in enumerate(self.components):
            if tuple(component.event_shape) != ():
                raise ValueError(f"Mixture component {index} must be univariate.")
            if tuple(component.batch_shape) != tuple(batch_shape):
                raise ValueError(
                    f"Mixture component {index} batch shape {tuple(component.batch_shape)} "
                    f"does not match gate batch shape {tuple(batch_shape)}."
                )
        super().__init__(batch_shape, validate_args=validate_args)

    @staticmethod
    def _expand_component(component: Distribution, batch_shape: torch.Size) -> Distribution:
        try:
            return component.expand(batch_shape)
        except NotImplementedError:
            # Distributions with a custom __init__ but no expand override
            # (e.g. the GluonTS StudentT subclass) make Distribution.expand
            # refuse to create the new instance itself. Supplying the instance
            # explicitly reuses the inherited parameter expansion; lazily
            # derived state is recomputed from the expanded parameters.
            return component.expand(
                batch_shape, _instance=component.__new__(type(component))
            )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Mixture, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mixture_logits = self.mixture_logits.expand(
            batch_shape + self.mixture_logits.shape[-1:]
        )
        new.components = [
            self._expand_component(component, batch_shape)
            for component in self.components
        ]
        super(Mixture, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def log_mixture_probs(self) -> Tensor:
        return torch.log_softmax(self.mixture_logits, dim=-1)

    @property
    def mixture_probs(self) -> Tensor:
        """Soft gate weights; useful for tracking expert collapse."""
        return torch.softmax(self.mixture_logits, dim=-1)

    @property
    def mean(self) -> Tensor:
        component_means = torch.stack(
            [component.mean for component in self.components], dim=-1
        )
        return (self.mixture_probs * component_means).sum(dim=-1)

    def component_log_probs(self, value: Tensor) -> Tensor:
        """Per-component log densities stacked on a trailing component axis."""
        return torch.stack(
            [component.log_prob(value) for component in self.components], dim=-1
        )

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return torch.logsumexp(
            self.log_mixture_probs + self.component_log_probs(value), dim=-1
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        sample_shape = torch.Size(sample_shape)
        with torch.no_grad():
            indices = Categorical(logits=self.mixture_logits).sample(sample_shape)
            component_samples = torch.stack(
                [component.sample(sample_shape) for component in self.components],
                dim=-1,
            )
            return torch.gather(
                component_samples, -1, indices.unsqueeze(-1)
            ).squeeze(-1)


class MixtureOutput(DistributionOutput):
    """Gated mixture head over multiple univariate distribution outputs.

    Each component keeps its own parameter projection and ``domain_map``; one
    extra linear projection produces the hidden-state-conditioned gate logits.
    """

    distr_cls: type = Mixture

    # NOTE: intentionally not `@validated()`; postponed annotations from
    # `from __future__ import annotations` leave pydantic with unresolvable
    # forward references for `List[DistributionOutput]`. As a consequence,
    # GluonTS serde does not record these constructor args: `serde.decode`
    # (used by `PyTorchPredictor.serialize`/`deserialize`) would rebuild
    # subclasses with default arguments and fail for this generic class.
    # That path is not used by this project — checkpoints pickle the head
    # instance inside `model_kwargs` via `torch.save`, which restores the
    # full component state exactly — so only pickle round-trips are supported.
    def __init__(self, components: List[DistributionOutput]) -> None:
        super().__init__()
        if len(components) < 2:
            raise ValueError("MixtureOutput requires at least two component outputs.")
        for index, component in enumerate(components):
            if tuple(component.event_shape) != ():
                raise ValueError(
                    f"MixtureOutput component {index} must be univariate "
                    f"(event_shape {component.event_shape})."
                )
        self.components = list(components)
        self.num_components = len(self.components)
        args_dim: Dict[str, int] = {"mixture_logits": self.num_components}
        for index, component in enumerate(self.components):
            for name, dim in component.args_dim.items():
                args_dim[f"component{index}_{name}"] = dim
        self.args_dim = args_dim

    def _component_arg_slices(self) -> List[int]:
        return [len(component.args_dim) for component in self.components]

    def domain_map(self, mixture_logits: Tensor, *component_args: Tensor):
        mapped: List[Tensor] = [mixture_logits]
        offset = 0
        for index, (component, num_args) in enumerate(
            zip(self.components, self._component_arg_slices())
        ):
            mapped_component = component.domain_map(
                *component_args[offset : offset + num_args]
            )
            if len(mapped_component) != num_args:
                raise RuntimeError(
                    f"MixtureOutput component {index} domain_map returned "
                    f"{len(mapped_component)} tensors, expected {num_args}."
                )
            mapped.extend(mapped_component)
            offset += num_args
        if offset != len(component_args):
            raise ValueError(
                f"MixtureOutput received {len(component_args)} component parameter "
                f"tensors, expected {offset}."
            )
        return tuple(mapped)

    def _base_distribution(self, distr_args) -> Mixture:
        mixture_logits = distr_args[0]
        components: List[Distribution] = []
        offset = 1
        for component, num_args in zip(self.components, self._component_arg_slices()):
            components.append(
                component._base_distribution(
                    tuple(distr_args[offset : offset + num_args])
                )
            )
            offset += num_args
        return Mixture(mixture_logits=mixture_logits, components=components)

    @property
    def event_shape(self) -> Tuple:
        return ()


class StudentTJohnsonSUMixtureOutput(MixtureOutput):
    """Gated two-component mixture of Student's t and Johnson SU."""

    def __init__(self) -> None:
        super().__init__(components=[StudentTOutput(), JohnsonSUOutput()])


class StudentTMixtureOutput(MixtureOutput):
    """Gated mixture of ``num_components`` Student's t distributions."""

    def __init__(self, num_components: int = 2) -> None:
        if num_components < 2:
            raise ValueError("StudentTMixtureOutput requires num_components >= 2.")
        super().__init__(
            components=[StudentTOutput() for _ in range(num_components)]
        )
