import numpy as np
import statistics as stat
import pandas as pd

def test(mi):
    result= np.array([m + 3* np.random.randn() for m in mi])
    est1 = result
    mi0 = 20 * np.ones(len(mi))
    est2= mi0 + max((1-1/sum((result-mi0)**2)),0) * (result-mi0)
    return np.array([stat.mean((est1-mi)**2),stat.mean((est2-mi)**2)]).reshape(1,2)

experiment=np.empty([1,2])
for i in range(10*6):
    experiment=np.concatenate((experiment,test([16,18,26])))

df_describe = pd.DataFrame(experiment,columns=["MSE1","MSE2"])
print(df_describe.describe().astype("float64").round(4))

#OUTPUT:
"""
MSE1     MSE2
count  61.0000  61.0000
mean    8.1530   8.0059
std     6.8755   6.7725
min     0.0000   0.0000
25%     4.1537   4.0499
50%     6.8799   6.6698
75%    10.6147  10.3617
max    35.5740  35.1391
"""
