from aftab.networks import PQNNetwork
from aftab.networks import DuelingNetwork
from aftab.networks import BootstrappedNetwork
from aftab.networks import BootstrappedDuelingNetwork
from aftab.networks import DistributionalNetwork
from aftab.networks import DistributionalDuelingNetwork
from aftab.networks import DistributionalBootstrappedDuelingNetwork

network_map = {
    "q": PQNNetwork,
    "duelling": DuelingNetwork,
    "bootstrapped": BootstrappedNetwork,
    "bootstrapped-duelling": BootstrappedDuelingNetwork,
    "distributional": DistributionalNetwork,
    "distributional-duelling": DistributionalDuelingNetwork,
    "distributional-bootstrapped-duelling": DistributionalBootstrappedDuelingNetwork,
    "bootstrapped-distributional-duelling": DistributionalBootstrappedDuelingNetwork,
    "d": DuelingNetwork,
    "bdd": DistributionalBootstrappedDuelingNetwork,
    "bd": BootstrappedDuelingNetwork,
    "dd": DistributionalDuelingNetwork,
    "b": BootstrappedNetwork,
}
