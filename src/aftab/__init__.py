from .Aftab import Aftab

from .constants import seeds
from .constants import seeds as SEEDS
from .constants import seeds as aftab_seeds

from .constants import atari_environments
from .constants import environments
from .constants import procgen_environments
from .constants import atari_environments as ATARI_ENVS
from .constants import procgen_environments as PROCGEN_ENVS
from .constants import environments as ENVS
from .constants import environments as aftab_environments

from importlib.metadata import version

try:
    __version__ = version("aftab")
except:
    __version__ = "development"

__all__ = [
    "Aftab",
    "seeds",
    "aftab_seeds",
    "aftab_environments",
    "environments",
    "atari_environments",
    "procgen_environments",
    "SEEDS",
    "ENVS",
    "ATARI_ENVS",
    "PROCGEN_ENVS",
]
