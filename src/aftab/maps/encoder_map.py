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
    "pqnhadamax": HadamaxNatureDQNEncoder,
    "hadamaxgammav1": HadamaxGammaEncoderValid,
    "gammahadamaxv1": HadamaxGammaEncoderValid,
    "hadamaxgammavalid": HadamaxGammaEncoderValid,
    "gammahadamaxvalid": HadamaxGammaEncoderValid,
    "hadamaxgammasame": HadamaxGammaEncoderSame,
    "gammahadamaxsame": HadamaxGammaEncoderSame,
    "hadamaxgammav2": HadamaxGammaEncoderSame,
    "gammahadamaxv2": HadamaxGammaEncoderSame,
    "hadamaxepsilon": HadamaxEpsilonEncoder,
    "epsilonhadamax": HadamaxEpsilonEncoder,
    "hadamaxzeta": HadamaxZetaEncoder,
    "zetahadamax": HadamaxZetaEncoder,
    "hadamaxdelta": HadamaxDeltaEncoder,
    "deltahadamax": HadamaxDeltaEncoder,
}
