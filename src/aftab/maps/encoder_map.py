from aftab.encoders import NatureDQNEncoder
from aftab.encoders import AlphaEncoder
from aftab.encoders import BetaEncoder
from aftab.encoders import GammaEncoder
from aftab.encoders import DeltaEncoder
from aftab.encoders import EpsilonEncoder
from aftab.encoders import EtaEncoder
from aftab.encoders import ZetaEncoder
from aftab.encoders import ThetaEncoder
from aftab.encoders import HadamaxNatureDQNEncoder
from aftab.encoders import HadamaxGammaEncoderSame
from aftab.encoders import HadamaxGammaEncoderValid
from aftab.encoders import HadamaxEpsilonEncoder
from aftab.encoders import HadamaxZetaEncoder
from aftab.encoders import HadamaxDeltaEncoder

encoder_map = {
    "nature": NatureDQNEncoder,
    "dqn": NatureDQNEncoder,
    "alpha": AlphaEncoder,
    "beta": BetaEncoder,
    "gamma": GammaEncoder,
    "delta": DeltaEncoder,
    "epsilon": EpsilonEncoder,
    "eta": EtaEncoder,
    "zeta": ZetaEncoder,
    "theta": ThetaEncoder,
    "hadamax": HadamaxNatureDQNEncoder,
    "dqnhadamax": HadamaxNatureDQNEncoder,
    "dqn-hadamax": HadamaxNatureDQNEncoder,
    "pqnhadamax": HadamaxNatureDQNEncoder,
    "pqn-hadamax": HadamaxNatureDQNEncoder,
    "gammahadamaxv1": HadamaxGammaEncoderValid,
    "gamma-hadamax-v1": HadamaxGammaEncoderValid,
    "gammahadamaxvalid": HadamaxGammaEncoderValid,
    "gamma-hadamax-valid": HadamaxGammaEncoderValid,
    "gammahadamaxsame": HadamaxGammaEncoderSame,
    "gamma-hadamax-same": HadamaxGammaEncoderSame,
    "gammahadamaxv2": HadamaxGammaEncoderSame,
    "gamma-hadamax-v2": HadamaxGammaEncoderSame,
    "epsilonhadamax": HadamaxEpsilonEncoder,
    "epsilon-hadamax": HadamaxEpsilonEncoder,
    "zeta-hadamax": HadamaxZetaEncoder,
    "zetahadamax": HadamaxZetaEncoder,
    "deltahadamax": HadamaxDeltaEncoder,
    "delta-hadamax": HadamaxDeltaEncoder,
}
