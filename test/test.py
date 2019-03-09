import numpy as np
import datetime
import time

N = 10000

with time.time() as tic:
    for _ in range(N):
        np.random.normal(0, 1)
    print(time.time() - tic)

print(datetime.datetime.now())
np.random.normal(0, 1, [N, 1])
print(datetime.datetime.now())
