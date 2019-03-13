# SimPricing
Provide a sim engine for pricing

仅需要重写SimBase._calc_payoff方法

使用时，可参照sim_sample.py中EuropeanOption和AsianOption。

### 2018.3.9

增加了对多种资产序列生成的支持，具体需要注意的是，在SimBase._calc_payoff中，返回的价格序列为np.array格式。

**使用时，请将numpy升级到最新版本**