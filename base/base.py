import abc
import numpy as np

from base.engine import SimEngine


class SimBase(object):
    def __init__(self, sim_engine, sim_times, r):
        """
        Base class for simulation
        :param sim_engine:  Type: SimEngine
        :param sim_times:   Type: int           Times of simulation
        :param r:           Type: float         risk-free interest rate, should be the same as r in sim_engine
                                                (if there is one)
        """
        if not isinstance(sim_engine, SimEngine):
            raise TypeError("Illegal input of sim_engine")

        self._sim_engine = sim_engine
        self._sim_times = sim_times

        if "r" in sim_engine.kwargs.keys():
            if not r == sim_engine.kwargs["r"]:
                raise ValueError("Illegal input of r, must be aligned with r in sim_engine!")

        self._r = r
        self._prc = None

    @property
    def prc(self):
        return self._prc

    @abc.abstractmethod
    def _calc_payoff(self, underlying_prc, sim_t, **kwargs):
        """
        Calculate discounted payoff of target derivative according to underlying_prc
        :param underlying_prc:       Type: float or series
        :param sim_t:                Type: float or iterable
        :return: discounted payoff
        :rtype: float
        """
        return

    def sim_exe(self, sim_t, **kwargs):
        """
        Execute simulation
        :param sim_t:     Type: Iterable or float
        :param kwargs:
        :return: expectation of price
        """
        prc_list = []
        for _ in range(self._sim_times):
            underlying_prc = self._sim_engine.prc_generator(sim_t)

            prc = self._calc_payoff(underlying_prc, sim_t, **kwargs)
            prc_list.append(prc)

        self._prc = np.mean(prc_list)
        return self._prc
