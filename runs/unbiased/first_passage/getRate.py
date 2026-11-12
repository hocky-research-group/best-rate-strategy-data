
from scipy import stats
from scipy.stats import gamma
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

res_t = []

for i in range(200):
    with open("run_"+str(i+1)+"/p.log",'r') as f:
        if len(f.readlines()) < 148:
            print("Run "+str(i+1)+" is still going.")
            continue
        else:
            print("Run "+str(i+1)+" is done.")
    with open("run_"+str(i+1)+"/p.log",'r') as f:
        for j in range(147):
            next(f)
        res_t.append(float(f.readline().split()[0])/1000000)

n_bins = 800
min_t = np.min(res_t)/10
max_t = np.max(res_t)*10
times  = np.logspace(np.log10(min_t),np.log10(max_t),n_bins)
P, bins = np.histogram(res_t,bins=times,density=True)

C = np.cumsum(P*(bins[1:]-bins[:-1]))

x = []
for i in range(len(C)):
    x.append(0.5*(bins[i]+bins[i+1]))

with open("distr.dat", 'w') as f:
    for i in range(len(P)):
        f.write(str(x[i]) + " " + str(C[i]) + "\n")

def ecdf(x, k):
    return 1-np.exp(-k*x)

k_o, pcov, info, mesg, ier = curve_fit(ecdf, np.float64(x), np.float64(C), full_output=True)

print(info['fvec'])
print('\n')
print("k = "+str(k_o)+"\n")
print("MFPT = "+str(statistics.mean(res_t)))

size = np.int64(len(res_t)*1e4)
rvs1 = gamma.rvs(1, scale=1/k_o, size=size)  # loc=min(time_list))
ks_stat, p = stats.ks_2samp(rvs1,res_t)

"""
with open("sanity.dat", 'w') as f:
    Ps, binss = np.histogram(rvs1,bins=times,density=True)
    Cs = np.cumsum(Ps*(binss[1:]-binss[:-1]))
    for i in range(len(C)):
        f.write(str(x[i]) + " " + str(Cs[i]) + "\n")
"""

print("ks_stat: "+str(ks_stat)+"; p = "+str(p))
