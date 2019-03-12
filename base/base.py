import abc
import numpy as np
import time
from multiprocessing import Pool
from itertools import repeat

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
        self._prc_list = []

        self._total_gen_time = None
        self._total_calc_time = None

    @property
    def prc(self):
        return self._prc

    @property
    def total_gen_time(self):
        return self._total_gen_time

    @property
    def total_calc_time(self):
        return self._total_calc_time

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

    def _single_sim_exe(self, sim_t, **kwargs):
        tic = time.time()
        underlying_prc = self._sim_engine.prc_generator(sim_t)
        self._total_gen_time += time.time() - tic

        tic = time.time()
        prc = self._calc_payoff(underlying_prc, sim_t, **kwargs)
        self._total_calc_time += time.time() - tic

        self._prc_list.append(prc)

    def sim_exe(self, sim_t, **kwargs):
        """
        Execute simulation
        :param sim_t:     Type: Iterable or float     Total times of simulation
        :param kwargs:
        :return: expectation of price
        """
        self._total_gen_time = 0
        self._total_calc_time = 0

        for _ in range(self._sim_times):
            self._single_sim_exe(sim_t, **kwargs)

        # full_sim = repeat(sim_t, self._sim_times)
        #
        # with Pool(2) as p:
        #     p.starmap(self._single_sim_exe, zip(full_sim, ))

        self._prc = np.mean(self._prc_list)
        return self._prc
