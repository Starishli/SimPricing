import numpy as np
import pandas as pd

from collections.abc import Iterable


class SimEngine(object):
    def __init__(self, method="GeoBrownian", s_0=100, **kwargs):
        """
        Currently support methods:
            GeoBrownian: Geometric Brownian Motion
        kwargs for:
            GeoBrownian: {"sigma": standard deviation, "r": risk-free expected payoff}
        :param method:    Type: String        To specify the method for generating price sequence.
        :param s_0:       Type: int or float  initial price of underlying asset
        :param kwargs:    Type: Dict
        """
        full_methods = ["GeoBrownian", ]
        full_kwarg_map = {"GeoBrownian": ["sigma", "r"], }

        if method not in full_methods:
            raise ValueError("Illegal input of method!")

        for key in kwargs.keys():
            if key not in full_kwarg_map[method]:
                raise ValueError("Illegal input of kwargs!")

        self._method = method
        self._kwargs = kwargs
        self._s_0 = s_0

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def s_0(self):
        return self._s_0

    def prc_generator(self, upper_t):
        """
        Generate price based on upper_t. Must keep in mind that upper_t is expressed in Year.
        :param upper_t:     Type: float or iterable object of float   expressed in Year
        :return:            Type: float or Series                     if upper_t is an iterable object, return a price
                                                                      series with index upper_t
        """
        s = self._s_0

        if isinstance(upper_t, Iterable):
            prc = [s, ]
            date = [0, ] + list(upper_t)
            t_diff = np.diff(upper_t, prepend=0)

            if self._method == "GeoBrownian":
                r = self._kwargs["r"]
                sigma = self._kwargs["sigma"]

                for t in t_diff:
                    epsilon = np.random.normal(0, 1)
                    s = s * np.exp((r - np.power(sigma, 2) / 2) * t + sigma * epsilon * np.sqrt(t))
                    prc.append(s)
            else:
                raise ValueError

            prc = pd.Series(data=prc, index=date)
        else:
            if self._method == "GeoBrownian":
                r = self._kwargs["r"]
                sigma = self._kwargs["sigma"]
                epsilon = np.random.normal(0, 1)

                prc = s * np.exp((r - np.power(sigma, 2) / 2) * upper_t + sigma * epsilon * np.sqrt(upper_t))
            else:
                raise ValueError

        return prc


if __name__ == "__main__":
    sim_engine = SimEngine(method="GeoBrownian", sigma=0.2, r=0.03)
    upper_t_ = range(0, 252, 1)
    upper_t_ = np.array(upper_t_) / 252

    prc_seq = sim_engine.prc_generator(upper_t=upper_t_)

    import matplotlib.pyplot as plt

    plt.plot(prc_seq)
    plt.show()



