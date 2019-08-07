# SimPricing
Provide a sim engine for pricing

you only need to rewrite SimBase._calc_payoff 

check EuropeanOption and AsianOption in sim_sample.py for reference

### 2019.3.9

support for multi asset simulation，return a np.array

**Make sure to upgrade your numpy to the latest version and install Numba**
