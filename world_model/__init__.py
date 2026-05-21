"""
InfiniteRace World Model — real-time latent diffusion for street-view navigation.
"""

from world_model.config import ModelConfig, TrainConfig, DistillConfig, InferenceConfig
from world_model.model import InfiniteRaceWorldModel

__all__ = [
    "InfiniteRaceWorldModel",
    "ModelConfig",
    "TrainConfig",
    "DistillConfig",
    "InferenceConfig",
]
