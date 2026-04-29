from thriftynet import ThriftyEncoder


class PsiEncoder(ThriftyEncoder):
    def __init__(self):
        super().__init__(filters=64, kernel_size=3, iterations=20)
