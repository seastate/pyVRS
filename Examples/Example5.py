import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.ion()

# test load the MorphMatrix object
filename = 'Freshwater/freshDM.dmgr'
with open(filename, 'rb') as handle:
    DM = pickle.load(handle)
    print(f'Loaded DataManager file {filename}')


# convert to array and average over inital angles
# the result is a n_a,n_b,n_e,n_s,n_v array of numpy vectors, each of which
# contains [avg_u,avg_v,avg_w,rel_avg_u,rel_avg_v,rel_avg_w]
sd = np.asarray(DM.sim_data).mean(axis=5)
# Get dimension lengths...
n_a,n_b,n_e,n_s,n_v,nvels = sd.shape

nsamples = n_a*n_b*n_e*n_s*n_v
print(f'nsamples = {nsamples}')


# extract velocity statistics; results are n_a,n_b,n_e,n_s,n_v
avg_us = sd[:,:,:,:,:,0]
avg_vs = sd[:,:,:,:,:,1]
avg_ws = sd[:,:,:,:,:,2]
avg_rel_us = sd[:,:,:,:,:,3]
avg_rel_vs = sd[:,:,:,:,:,4]
avg_rel_ws = sd[:,:,:,:,:,5]


alpha_set = DM.MM.pars.alpha_set
beta_set = DM.MM.pars.beta_set
eta_set = DM.MM.pars.eta_set
Vcil_set = DM.SM.spars.Vcil_set
shear_set = DM.SM.spars.shear_set
XEinits = DM.SM.spars.XEinits

print('alpha_set = ',alpha_set)
print('beta_set = ',beta_set)
print('eta_set = ',eta_set)
print('Vcil_set = ',Vcil_set)
print('shear_set = ',shear_set)
print('XEinits = ',XEinits)

avg_rel_ws[:,:,:,:,2].max()
avg_rel_ws[:,:,:,:,2].min()


# Generate a matrix of plots
# set ciliary velocity
i_v = 2
levels5 = np.linspace(-5.5,1.,51)
#levels5 = np.linspace(-0.125,0.2125,51)

fig5, axes5 = plt.subplots(n_e,n_s,layout='constrained')
plt_count = 0

for i_s,shear in enumerate(shear_set):
    for i_e,eta in enumerate(eta_set):
        test5 = avg_rel_ws[:,:,i_e,i_s,2]
        #axes5[i_e,i_s].contourf(beta_set, alpha_set, test5, levels5)
        #axes5[i_e,i_s].contourf(beta_set, alpha_set, test5, levels5, cmap=plt.cm.plasma)
        axes5[i_e,i_s].contourf(beta_set, alpha_set, test5, levels5, cmap=plt.cm.jet)
        #axes5[i_e,i_s].contourf(beta_set, alpha_set, test5, levels5, cmap=plt.cm.bone)





avg_rel_ws[:,:,1,0,2].max()
avg_rel_ws[:,:,1,0,2].min()

avg_rel_ws[:,:,1,1,2].max()
avg_rel_ws[:,:,1,1,2].min()





#================================================
# An example plot of rel_w across alpha, beta 

#test = sd[:,:,0,2,2,0].squeeze()
#test = avg_rel_ws[:,:,0,0,0]

levels = np.linspace(-5,1,41)

test1 = avg_rel_ws[:,:,0,0,0]
fig1, ax1 = plt.subplots(layout='constrained')
#CS = ax1.contourf(beta_set, alpha_set, test1, 10, cmap=plt.cm.bone)
CS = ax1.contourf(beta_set, alpha_set, test1, levels)#, cmap=plt.cm.bone)
cbar = fig1.colorbar(CS)
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'$\beta$')

test2 = avg_rel_ws[:,:,0,0,2]
fig2, ax2 = plt.subplots(layout='constrained')
#CS2 = ax2.contourf(beta_set, alpha_set, test2, 10, cmap=plt.cm.bone)
CS2 = ax2.contourf(beta_set, alpha_set, test2, levels)#, cmap=plt.cm.bone)
cbar2 = fig2.colorbar(CS2)
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$\beta$')
