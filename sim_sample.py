import numpy as np

from base.engine import SimEngine
from base.base import SimBase


class EuropeanOption(SimBase):
    def __init__(self, sim_engine, sim_times, r):
        super().__init__(sim_engine, sim_times, r)

    def _calc_payoff(self, underlying_prc, sim_t, **kwargs):
        strike = kwargs["strike"]
        prc = max(underlying_prc[0][0] - strike, 0) * np.exp(-self._r * sim_t)
        return prc


class AsianOption(SimBase):
    def __init__(self, sim_engine, sim_times, r):
        super().__init__(sim_engine, sim_times, r)

    def _calc_payoff(self, underlying_prc, sim_t, **kwargs):
        strike = kwargs["strike"]
        final_prc = np.mean(underlying_prc)

        prc = max(final_prc - strike, 0) * np.exp(-self._r * sim_t[-1])
        return prc


if __name__ == "__main__":
    r_ = 0.03
    sim_times_ = 100000
    sim_t_1 = 20 / 365
    sim_t_2 = np.array([5, 10, 15, 20]) / 365

    sim_engine_ = SimEngine(sigma=0.3, r=r_)

    eu_opt = EuropeanOption(sim_engine_, sim_times_, r_)
    eu_p = eu_opt.sim_exe(sim_t_1, strike=100)
    print(eu_p)
    print("=================")
    print(eu_opt.total_gen_time)
    print(eu_opt.total_calc_time)

    print("+++++++++++++++++++++++")

    as_opt = AsianOption(sim_engine_, sim_times_, r_)
    as_p = as_opt.sim_exe(sim_t_2, strike=100)
    print(as_p)
    print("=================")
    print(as_opt.total_gen_time)
    print(as_opt.total_calc_time)



