from ..networks import PQNNetwork
from ..networks import DuellingNetwork
from ..networks import EnsembleNetwork
from ..networks import EnsembleDuellingNetwork
from ..networks import DistributionalPQNNetwork
from ..networks import DistributionalDuellingNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "ensemble": EnsembleNetwork,
    "ensemble-duelling": EnsembleDuellingNetwork,
    "distributional": DistributionalPQNNetwork,
    "distributional-duelling": DistributionalDuellingNetwork,
}
network_map = networks_map
