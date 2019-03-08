import abc
import numpy as np

from base.engine import SimEngine


class SimBase(object):
    def __init__(self, sim_engine, sim_times):
        """
        Base class for simulation
        :param sim_engine:  Type: SimEngine
        :param sim_times:   Type: Int           Times of simulation
        """
        if not isinstance(sim_engine, SimEngine):
            raise TypeError("Illegal input of sim_engine")

        self.sim_engine = sim_engine
        self.sim_times = sim_times
        self.r = self.sim_engine.kwargs["r"]

        self.prc = None

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
        for _ in range(self.sim_times):
            underlying_prc = self.sim_engine.prc_generator(sim_t)

            prc = self._calc_payoff(underlying_prc, sim_t, **kwargs)
            prc_list.append(prc)

        return np.mean(prc_list)
