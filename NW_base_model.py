# -*- coding: utf-8 -*-

import numpy as np
import collections
import matplotlib.pylab as plt

firms_set = {2: [139.58, 0.00143, 0.0287], 4:[89.70, 0.00112, 0.0223],
         8:[48.85,0.00102,0.0205], 16:[25.34, 0.00099, 0.0197],
         32:[12.82, 0.00097, 0.0194]}

firms_set = collections.OrderedDict(sorted(firms_set.items()))

time = range (1, 102)


def init (A_init, K_init, n_firms_init, im_init, in_init):
    global A_i, K_i, im_i, in_i, n_firms
    n_firms = n_firms_init
    A_i =  A_init
    K_i = K_init
    im_i = im_init
    in_i = in_init
    in_firms = {x: [] for x in range (n_firms / 2)}
    im_firms = {x: [] for x in range (n_firms / 2)}
    [in_firms[x].append([A_i, Q_i, K_i, 0, im_i, in_i]) for x in in_firms]
    [im_firms[x].append([A_i, Q_i, K_i, 0, im_i]) for x in in im_firms]
    #firms = in_firms + im_firms
    
    
def Q_updates(in_firms, im_firms):
    global Q_tot
    Q_tot = 0
    for i in in_firms:
        in_firms[i][0][1] = in_firms[i][0][0]*in_firms[i][0][2] 
        Q_tot = in_firms[i][0][1] + Q_tot
    for i in im_firms:
        in_firms[i][0][1] = in_firms[i][0][0]*in_firms[i][0][2] 
        Q_tot = in_firms[i][0][1] + Q_tot
        
    
def P_update (Q_tot):
    global P
    
    P = 67./Q_tot

    
def profit_updates (K_i, A_i, in_i, im_i, P):
    global pi
    in_firm = P*A_i - 0.16 - in_i - im_i
    im_firm = P*A_i - 0.16 - im_i
    
def A_updates (in_i, im_i):
    As = [] 
    if np.random.random() < in_i:
        lamb = 0.16 + 0.01*t
        As.append(np.exp(np.random.normal(lamb, 0.05)))
    if np.random.random() < im_i:
        As.append(max(max(A_infimrs), max(A_imfirms)))
    A_i = max(As)
    

def I_updates ():
    rho = P*A_i/(0.16)
    s = Q_i/Q_tot
    I_i = max(0, min((1.03 - (2*s)/(rho*(2-2*s))), 0.03 + pi_i if pi_i <= 0 else 0.03 + 2*pi_i))
    
def K_update():
    K_i = I*K_i + 0.97*K_i
    

for n in firms_set:
    init(A_init=0.16, K_init=firms_set[n][0], n_firms_init=n, im_init=firms_set[n][1], , in_init=firms_set[n][2])
    
    Q_updates(in_firms=in_firms, im_firms=im_firms)
    for t in time:
        
    
    
    
    
