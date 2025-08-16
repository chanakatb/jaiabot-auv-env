import torch

from dataclasses import MISSING
from collections.abc import Sequence

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, NoiseModel, NoiseModelCfg, NoiseModelWithAdditiveBias, NoiseCfg, gaussian_noise

class AsymmetricNoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.

    """

    def __init__(self, num_envs: int, noise_model_cfg: NoiseModelCfg, device: str):
        super().__init__(num_envs, noise_model_cfg)
        self._device = device
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, noise_model_cfg.dims), device=self._device)

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        r"""Apply the noise + bias.

        Args:
            data: The data to apply the noise to, which is a tensor of shape (num_envs, \*data_shape).
        """
        return super().apply(data) + self._bias

    def reset(self, env_ids: Sequence[int]):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for.
        """
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

@configclass
class AsymmetricNoiseModelWithAdditiveBiasCfg(NoiseModelCfg):
    """Configuration for an additive gaussian noise with bias model."""

    class_type: type = AsymmetricNoiseModelWithAdditiveBias

    # Provide default values instead of MISSING
    bias_noise_cfg: NoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=1.0,
        operation="add"
    )

    dims: int = 1

def asymmetric_gaussian_noise(data: torch.Tensor, cfg: NoiseCfg) -> torch.Tensor:
    """Gaussian noise."""
    mean = cfg.mean.repeat(data.shape[0],1) if hasattr(cfg.mean, 'repeat') else torch.full_like(data, cfg.mean)
    std = cfg.std.repeat(data.shape[0],1) if hasattr(cfg.std, 'repeat') else torch.full_like(data, cfg.std)

    if cfg.operation == "add":
        return data + mean + std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (mean + std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return mean + std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")

@configclass
class AsymmetricGaussianNoiseCfg(NoiseCfg):
    """Configuration for an additive gaussian noise term."""

    func = asymmetric_gaussian_noise

    # Provide default values instead of MISSING
    mean: torch.Tensor | float = 0.0
    """The mean of the noise. Defaults to 0.0."""
    std: torch.Tensor | float = 1.0
    """The standard deviation of the noise. Defaults to 1.0."""