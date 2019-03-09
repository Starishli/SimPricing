import time
import numpy as np
import pandas as pd

from collections.abc import Iterable


class SimEngine(object):
    def __init__(self, method="GeoBrownian", s_0=100, **kwargs):
        """
        Currently support methods:
            GeoBrownian: Geometric Brownian Motion
        kwargs for:
            GeoBrownian: {"sigma": standard deviation, float or list
                          "r": risk-free expected payoff, float
                          "rho": correlation matrix, numpy.matrix
                          }
        :param method:    Type: String        To specify the method for generating price sequence.
        :param s_0:       Type: int or float  initial price of underlying asset
        :param kwargs:    Type: Dict
        """
        full_methods = ["GeoBrownian", ]
        full_kwarg_map = {"GeoBrownian": ["sigma", "r", "rho"], }

        if method not in full_methods:
            raise ValueError("Illegal input of method!")

        for key in kwargs.keys():
            if key not in full_kwarg_map[method]:
                raise ValueError("Illegal input of kwargs!")

        self._method = method
        self._kwargs = kwargs
        self._s_0 = s_0
        self._n_dim = None

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def s_0(self):
        return self._s_0

    @property
    def n_dim(self):
        if self._method == "GeoBrownian":
            if isinstance(self.kwargs["sigma"], list):
                return len(self.kwargs["sigma"])
            else:
                return 1
        else:
            raise ValueError

    def prc_generator(self, upper_t):
        """
        Generate price based on upper_t. Must keep in mind that upper_t is expressed in Year.
        :param upper_t:     Type: float or iterable object of float   expressed in Year
        :return:            Type: DataFrame   columns: upper_t, seq_N (N start from 0)
        """
        if not isinstance(upper_t, Iterable):
            upper_t = [upper_t, ]

        if self._method == "GeoBrownian":
            t_s = time.time()

            s_array = np.fromiter([self.s_0, ] * self.n_dim, dtype="float64")

            prc = []
            t_diff = np.diff(upper_t, prepend=0)

            r = self.kwargs["r"]

            sigma_raw = self.kwargs["sigma"]
            sigma_array = np.fromiter(sigma_raw if isinstance(sigma_raw, Iterable) else [sigma_raw, ],
                                      dtype="float64")

            if self.n_dim > 1:
                rho = self.kwargs["rho"]
                upper_r = np.linalg.cholesky(rho)

                if not isinstance(rho, np.matrix):
                    raise ValueError("Illegal input type of rho")

                if rho.shape[0] != rho.shape[1] or rho.shape[0] != self.n_dim:
                    raise ValueError("Illegal input dimension of rho")
            else:
                upper_r = 1

            time_1 = time.time() - t_s
            t_s = time.time()

            for t in t_diff:
                x = np.random.normal(0, 1, [self.n_dim, 1])
                epsilon_array = np.fromiter(upper_r * x, dtype="float64")

                s_array = s_array * np.exp((r - np.power(sigma_array, 2) / 2) * t
                                           + sigma_array * epsilon_array * np.sqrt(t))
                prc.append(s_array)

            prc = np.array(prc)

            time_2 = time.time() - t_s
        else:
            raise ValueError

        return prc, time_1, time_2


if __name__ == "__main__":
    # rho_ = np.matrix([[1, -0.999], [-0.999, 1]])
    # sigma_ = [0.2, 0.3]

    sigma_ = 0.4

    sim_engine = SimEngine(method="GeoBrownian", sigma=sigma_, r=0.03)
    upper_t_ = range(1, 252, 1)
    upper_t_ = np.array(upper_t_) / 252

    prc_seq = sim_engine.prc_generator(upper_t=upper_t_)

    import matplotlib.pyplot as plt

    prc_seq.plot()
    plt.show()



