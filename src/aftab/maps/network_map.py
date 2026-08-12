from aftab.networks import PQNNetwork
from aftab.networks import DuelingNetwork
from aftab.networks import BootstrappedNetwork
from aftab.networks import BootstrappedDuelingNetwork
from aftab.networks import DistributionalNetwork
from aftab.networks import DistributionalDuelingNetwork
from aftab.networks import DistributionalBootstrappedDuelingNetwork

network_map = {
    "q": PQNNetwork,
    "dueling": DuelingNetwork,
    "bootstrapped": BootstrappedNetwork,
    "bootstrapped-dueling": BootstrappedDuelingNetwork,
    "distributional": DistributionalNetwork,
    "distributional-dueling": DistributionalDuelingNetwork,
    "distributional-bootstrapped-dueling": DistributionalBootstrappedDuelingNetwork,
    "bootstrapped-distributional-dueling": DistributionalBootstrappedDuelingNetwork,
    "d": DuelingNetwork,
    "bdd": DistributionalBootstrappedDuelingNetwork,
    "bd": BootstrappedDuelingNetwork,
    "dd": DistributionalDuelingNetwork,
    "b": BootstrappedNetwork,
}
