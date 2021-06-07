
import numpy as np
import matplotlib.pyplot as plt
import ot
import fpot
import ProximalPointOT
import matplotlib
matplotlib.rc('xtick',labelsize = 20)
matplotlib.rc('ytick',labelsize = 20)

n = 100
x = np.arange(n,dtype = np.float64)

p1 = 0.55 * ot.datasets.get_1D_gauss(n,20,8) + 0.45 * ot.datasets.get_1D_gauss(n,70,9)
p2 = 0.55 * ot.datasets.get_1D_gauss(n,35,9) + 0.45 * ot.datasets.get_1D_gauss(n,55,5)

plt.plot(x,p1,'o-',color='blue')
plt.plot(x,p2,'o-',color='red')
plt.tight_layout()
plt.show()

C = ot.utils.dist0(n)
print(C)
C/=C.max()

T_emd = ot.emd(p1,p2,C)
ground_truth = np.sum(T_emd*C)

maxiter = 2000
beta_list = [0.001,0.01,0.1,1]
inner_maxiter = 1
use_path = True

### proximal point OT
T_pp_list  = []
loss_pp_list = []

for beta in beta_list:
    # T, loss = fpot.fpot_wd(p1, p2, C, beta=beta, maxiter=maxiter, inner_maxiter=inner_maxiter, use_path=use_path)
    T,loss = ProximalPointOT.ProximalPointOT(p1,p2,C,beta = beta, maxiter = maxiter, inner_maxiter = inner_maxiter, use_path = use_path)
    loss_pp_list.append(loss)
    T_pp_list.append(T)
### colormap
import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('Reds')
new_cmap = truncate_colormap(cmap, 0., 0.8)


f,axarr = plt.subplots(1,len(beta_list),figsize = (9,3))
for i, beta in enumerate(beta_list):
    axarr[i].imshow(T_pp_list[i],cmap = new_cmap)
    axarr[i].imshow(T_emd,cmap = plt.get_cmap('binary'),alpha = 0.7)
    axarr[i].xaxis.set_ticks([])
    axarr[i].yaxis.set_ticks([])
    axarr[i].set_title(r'$\beta$ = ' + str(beta), fontsize = 20)
plt.show()

### EOT
T_eot_list  = []
loss_eot_list = []

for beta in beta_list:
    T,loss = ProximalPointOT.EOT(p1,p2,C,beta = beta, maxiter = maxiter)
    loss_eot_list.append(loss)
    T_eot_list.append(T)

f,axarr = plt.subplots(1,len(beta_list),figsize = (9,3))
for i, beta in enumerate(beta_list):
    axarr[i].imshow(T_eot_list[i],cmap = new_cmap)
    axarr[i].imshow(T_emd,cmap = plt.get_cmap('binary'),alpha = 0.7)
    axarr[i].xaxis.set_ticks([])
    axarr[i].yaxis.set_ticks([])
    axarr[i].set_title(r'$\beta$ = ' + str(beta), fontsize = 20)
plt.show()