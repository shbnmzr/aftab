from .exceptions import InvalidEncoderIdentifier


def panic_if_encoder_identifier_is_illegal(encoder_name: str):
    if encoder_name not in [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "zeta",
        "eta",
        "epsilon",
        "theta",
    ]:
        raise InvalidEncoderIdentifier(encoder_name)
