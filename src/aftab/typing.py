import torch
from typing import Type
from typing import Literal, Annotated, TypeAlias

ModuleType: TypeAlias = Type[torch.nn.Module]

EncoderStringType: TypeAlias = Annotated[str, "must be a valid encoder key"]

OptimizerStringType: TypeAlias = Literal["adam", "adamw" "radam", "nadam"]

NetworkStringType: TypeAlias = Literal[
    "q",
    "dueling",
    "bootstrapped",
    "bootstrapped-dueling",
    "distributional",
    "distributional-dueling",
    "distributional-bootstrapped-dueling",
]
