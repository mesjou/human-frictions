from agents.bank import CentralBank


def test_set_interest_rate():
    bank = CentralBank(
        inflation_target=0.02, natural_unemployment=0.0, natural_interest=0.0, phi_unemployment=0.1, phi_inflation=0.2,
    )
    interest_rate = bank.set_interest_rate(unemployment=0.0, inflation=0.02)
    assert interest_rate == 1.02

    interest_rate = bank.set_interest_rate(unemployment=0.1, inflation=0.02)
    assert 1.0 <= interest_rate < 1.02

    interest_rate = bank.set_interest_rate(unemployment=0.0, inflation=0.01)
    assert 1.0 <= interest_rate < 1.02

    interest_rate = bank.set_interest_rate(unemployment=0.0, inflation=0.03)
    assert interest_rate > 1.02

    bank_1 = CentralBank(
        inflation_target=0.5, natural_unemployment=0.0, natural_interest=0.0, phi_unemployment=0.1, phi_inflation=0.2,
    )
    interest_rate = bank_1.set_interest_rate(unemployment=0.0, inflation=0.0)
    assert interest_rate < 1.5


if __name__ == "__main__":
    test_set_interest_rate()
