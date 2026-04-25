from ..networks import (
    PQNNetwork,
    DuellingNetwork,
    DistributionalPQNNetwork,
    DistributionalDuellingNetwork,
)

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "distributional": DistributionalPQNNetwork,
    "distributional-duelling": DistributionalDuellingNetwork,
}
network_map = networks_map
