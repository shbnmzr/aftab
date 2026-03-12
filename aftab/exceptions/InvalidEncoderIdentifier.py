class InvalidEncoderIdentifier(Exception):
    def __init__(self, illegal_identifier: str):
        message = f"{illegal_identifier} is not valid. Choose among alpha, beta, gamma, epsilon, delta, eta, zeta, theta."
        super().__init__(message)
