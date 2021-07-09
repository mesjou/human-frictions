class CentralBank(object):
    def __init__(
        self,
        inflation_target: float = 0.02,
        natural_unemployment: float = 0.0,
        natural_interest: float = 0.0,
        phi_unemployment: float = 0.1,
        phi_inflation: float = 0.2,
    ):
        assert isinstance(inflation_target, float)
        assert inflation_target >= 0.0
        self.inflation_target = inflation_target

        assert isinstance(natural_unemployment, float)
        assert natural_unemployment >= 0.0
        self.natural_unemployment = natural_unemployment

        assert isinstance(phi_unemployment, float)
        assert phi_unemployment > 0.0
        self.phi_unemployment = phi_unemployment

        assert isinstance(phi_inflation, float)
        assert phi_inflation > 0.0
        self.phi_inflation = phi_inflation

        assert isinstance(natural_interest, float)
        assert natural_interest >= 0.0
        self.natural_interest = natural_interest

    def set_interest_rate(self, unemployment: float, inflation: float):
        interest_rate = (1 + self.inflation_target) * (1 + self.natural_interest)
        interest_rate *= ((1 + inflation) / (1 + self.inflation_target)) ** self.phi_inflation
        interest_rate *= ((1 + self.natural_unemployment) / (1 + unemployment)) ** self.phi_unemployment
        return interest_rate
