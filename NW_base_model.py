# -*- coding: utf-8 -*-

import numpy as np
import collections
import matplotlib.pylab as plt


def init (A_init, K_init, n_firms_init, im_init, in_init, pi_init, q_init, I_init):
    global A_i, K_i, im_i, in_i, n_firms, in_firms, im_firms, I_i, q_i, pi_i
    n_firms = n_firms_init
    A_i =  A_init
    K_i = K_init
    im_i = im_init
    in_i = in_init
    I_i = I_init
    q_i = q_init
    pi_i = pi_init
    in_firms = {x: [] for x in range (n_firms / 2)}
    im_firms = {x: [] for x in range (n_firms / 2)}
    [in_firms[x].append([A_i, q_i, K_i, pi_i, im_i, in_i, I_i]) for x in in_firms]
    [im_firms[x].append([A_i, q_i, K_i, pi_i, im_i, I_i]) for x in im_firms]
    
    
def Q_updates(in_firms, im_firms):
    global Q_tot
    Q_tot = 0
    for i in in_firms:
        in_firms[i][0][1] = in_firms[i][0][0]*in_firms[i][0][2] 
        Q_tot = in_firms[i][0][1] + Q_tot
    for i in im_firms:
        im_firms[i][0][1] = im_firms[i][0][0]*im_firms[i][0][2] 
        Q_tot = im_firms[i][0][1] + Q_tot
        
    #print Q_tot
    
        
    
def P_update (Q_tot):
    global P
    P = 67./Q_tot

    
def pi_updates (in_firms, im_firms, P):
    for i in im_firms:
        im_firms[i][0][3] = P*im_firms[i][0][0] - 0.16 - im_firms[i][0][4]
    for i in in_firms:
        in_firms[i][0][3] = P*in_firms[i][0][0] - 0.16 - in_firms[i][0][5] -\
        in_firms[i][0][4] 
    
    
def A_updates (im_firms, in_firms):
    for i in in_firms:
        As = []
        if np.random.random() < 0.125*in_firms[i][0][5]*in_firms[i][0][2]:
            lamb = 0.16 + 0.01*t
            As.append(np.exp(np.random.normal(lamb, 0.05)))
        if np.random.random() < 1.25*in_firms[i][0][4]*in_firms[i][0][2]:
            As.append(max(max([in_firms[x][0][0] for x in in_firms]),\
                          max([im_firms[x][0][0] for x in im_firms])))
        As.append(in_firms[i][0][0])
        in_firms[i][0][0] = max(As)
    for i in im_firms:
        As = []
        if np.random.random() < 1.25*im_firms[i][0][4]*im_firms[i][0][2]:
            As.append(max(max([in_firms[x][0][0] for x in in_firms]),\
                          max([im_firms[x][0][0] for x in im_firms])))
        As.append(in_firms[i][0][0])
        im_firms[i][0][0] = max(As)
        

def I_updates (in_firms, im_firms, Q_tot, P):
    for i in in_firms:
        rho = P*in_firms[i][0][0]/(0.16)
        s = in_firms[i][0][1]/(Q_tot)
        pi = in_firms[i][0][3]
        in_firms[i][0][6] = max(0, min((1.03 - (2. -s)/(rho*(2-(2.*s)))), 0.03 +\
                pi if pi <= 0 else 0.03 + 2*pi))
        
    for i in im_firms:
        rho = P*im_firms[i][0][0]/(0.16)
        s = im_firms[i][0][1]/(Q_tot)
        pi = im_firms[i][0][3]
        im_firms[i][0][5] = max(0, min((1.03 - (2. -s)/(rho*(2-(2.*s)))), 0.03 +\
                pi if pi <= 0 else 0.03 + 2*pi))
                
    
def K_update(in_firms, im_firms):
    for i in in_firms:
        in_firms[i][0][2] = in_firms[i][0][2]*in_firms[i][0][6] +\
        0.97*in_firms[i][0][2] 
        
    for i in im_firms:
        im_firms[i][0][2] = im_firms[i][0][2]*im_firms[i][0][5] +\
        0.97*im_firms[i][0][2] 
    
################################## Set the Conditions #########################
firms_set = {2: [139.58, 0.00143, 0.0287], 4:[89.70, 0.00112, 0.0223],
         8:[48.85,0.00102,0.0205], 16:[25.34, 0.00099, 0.0197],
         32:[12.82, 0.00097, 0.0194]}
firms_set = collections.OrderedDict(sorted(firms_set.items()))
time = range (1, 102)
mcs = range (1, 20)
###############################################################################

prices = []
AVG_As = []
Best_As = []


i = 1
while i <= 20:
    price = []
    AVG_A = []
    Best_A =[]
    for n in firms_set: ### Loop que decide quantas firmas
        init(A_init=0.16, K_init=firms_set[n][0], n_firms_init=n,\
             im_init=firms_set[n][1], in_init=firms_set[n][2],\
             pi_init=0, q_init=0, I_init=0)
        
        Q_updates(in_firms=in_firms, im_firms=im_firms)
        P_update(Q_tot=Q_tot)
        pi_updates(in_firms=in_firms, im_firms=im_firms, P=P)
        # A_updates(im_firms=im_firms, in_firms=in_firms)
        I_updates(in_firms=in_firms, im_firms=im_firms, Q_tot=Q_tot, P=P)
        K_update(in_firms=in_firms, im_firms=im_firms)
        
        for t in time: ### loop 
            A_updates(im_firms=im_firms, in_firms=in_firms)
            Q_updates(in_firms=in_firms, im_firms=im_firms)
            P_update(Q_tot=Q_tot)
            pi_updates(in_firms=in_firms, im_firms=im_firms, P=P)
            #A_updates(im_firms=im_firms, in_firms=in_firms)
            I_updates(in_firms=in_firms, im_firms=im_firms, Q_tot=Q_tot, P=P)
            K_update(in_firms=in_firms, im_firms=im_firms)
        price.append(P)
        AVG_A.append(np.mean([np.mean([in_firms[x][0][0] for x in in_firms]),\
                     np.mean([im_firms[x][0][0] for x in im_firms])]))
        Best_A.append(max(max([in_firms[x][0][0] for x in in_firms]),\
                          max([in_firms[x][0][0] for x in im_firms])))
    prices.append(price)
    AVG_As.append(AVG_A)
    Best_As.append(Best_A)
    
    i = i+1
    
labels = ["prices", "AVG_As", "Best_As"]   
metrics = [prices, AVG_As, Best_As]

for metric in metrics:
    var_plot = []
    for i in range(len(firms_set)):
        vars2_32 = np.mean([metric[x][i] for x in range (len(metric))])
        var_plot.append(vars2_32)
    plt.title(labels[metrics.index(metric)])    
    plt.plot(firms_set.keys(), var_plot)
    plt.show()
