from ..networks import PQNNetwork, DuellingNetwork, FQFNetwork, DuellingFQFNetwork

networks_map = {
    "q": PQNNetwork,
    "duelling": DuellingNetwork,
    "fqf": FQFNetwork,
    "fqf-duelling": DuellingFQFNetwork,
    "duelling-fqf": DuellingFQFNetwork,
}
network_map = networks_map
