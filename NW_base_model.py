# -*- coding: utf-8 -*-

import numpy as np
import collections
import matplotlib.pylab as plt


def init (A_init, K_init, n_firms_init, im_init, in_init):
    global A_i, K_i, im_i, in_i, n_firms, in_firms, im_firms
    n_firms = n_firms_init
    A_i =  A_init
    K_i = K_init
    im_i = im_init
    in_i = in_init
    in_firms = {x: [] for x in range (n_firms / 2)}
    im_firms = {x: [] for x in range (n_firms / 2)}
    [in_firms[x].append([A_i, 0, K_i, 0, im_i, in_i, 0]) for x in in_firms]
    [im_firms[x].append([A_i, 0, K_i, 0, im_i, 0]) for x in im_firms]
    
    
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

    
def pi_updates (in_firms, im_firms, P):
    global pi 
    for i in im_firms:
        im_firms[i][0][3] = P*im_firms[i][0][0] - 0.16 - im_firms[i][0][4]
    for i in in_firms:
        im_firms[i][0][3] = P*im_firms[i][0][0] - 0.16 - im_firms[i][0][5] -\
        im_firms[i][0][4] 
    
    
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
        pi_i = in_firms[i][0][3]
        in_firms[i][0][6] = max(0, min((1.03 - (2*s)/(rho*(2-2*s))), 0.03 +\
                pi_i if pi_i <= 0 else 0.03 + 2*pi_i))
        
    for i in im_firms:
        rho = P*im_firms[i][0][0]/(0.16)
        s = im_firms[i][0][1]/(Q_tot)
        pi_i = im_firms[i][0][3]
        im_firms[i][0][5] = max(0, min((1.03 - (2*s)/(rho*(2-2*s))), 0.03 +\
                pi_i if pi_i <= 0 else 0.03 + 2*pi_i))
                
    
def K_update(in_firms, im_firms):
    for i in in_firms:
        in_firms[i][0][2] = in_firms[i][0][2]*in_firms[i][0][6] +\
        0.97*in_firms[i][0][2] 
        
    for i in im_firms:
        im_firms[i][0][2] = im_firms[i][0][2]*im_firms[i][0][5] +\
        0.97*in_firms[i][0][2] 
    
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
while i <= 10:
    price = []
    AVG_A = []
    Best_A =[]
    for n in firms_set:
        init(A_init=0.16, K_init=firms_set[n][0], n_firms_init=n,\
             im_init=firms_set[n][1], in_init=firms_set[n][2])
        
        Q_updates(in_firms=in_firms, im_firms=im_firms)
        P_update(Q_tot=Q_tot)
        ### a partir daqui atualizando as variaei para o periodo seguinte
        pi_updates(in_firms=in_firms, im_firms=im_firms, P=P)
        A_updates(im_firms=im_firms, in_firms=in_firms)
        I_updates(in_firms=in_firms, im_firms=im_firms, Q_tot=Q_tot, P=P)
        K_update(in_firms=in_firms, im_firms=im_firms)
        
        for t in time:
            Q_updates(in_firms=in_firms, im_firms=im_firms)
            P_update(Q_tot=Q_tot)
            pi_updates(in_firms=in_firms, im_firms=im_firms, P=P)
            A_updates(im_firms=im_firms, in_firms=in_firms)
            I_updates(in_firms=in_firms, im_firms=im_firms, Q_tot=Q_tot, P=P)
            K_update(in_firms=in_firms, im_firms=im_firms)
        price.append(P)
        AVG_A.append(np.mean([np.mean([in_firms[x][0][0] for x in in_firms]),\
                     np.mean([in_firms[x][0][0] for x in im_firms])]))
        Best_A.append(max(max([in_firms[x][0][0] for x in in_firms]),\
                          max([in_firms[x][0][0] for x in im_firms])))
    prices.append(price)
    AVG_As.append(AVG_A)
    Best_As.append(Best_A)
    
    """
    AVG_A.append(np.mean(np.mean([in_firms[x][0][0] for x in in_firms]),
                 np.mean([im_firms[x][0][0] for x in im_firms])))
    Best_practice.append(max(max([in_firms[x][0][0] for x in in_firms]),\
                             max([im_firms[x][0][0] for x in im_firms])))
        """
    i = i+1
    
p2 = np.mean([prices[x][0] for x in range(len(prices))])
p4 = np.mean([prices[x][1] for x in range(len(prices))])
p8 = np.mean([prices[x][2] for x in range(len(prices))])
p16 = np.mean([prices[x][3] for x in range(len(prices))])
p32 = np.mean([prices[x][4] for x in range(len(prices))])

plt.plot([2,4,8,16,32], [p2,p4,p8,p16,p32])
plt.show()




A2 = np.mean([AVG_As[x][0] for x in range(len(AVG_As))])
A4 = np.mean([AVG_As[x][1] for x in range(len(AVG_As))])
A8 = np.mean([AVG_As[x][2] for x in range(len(AVG_As))])
A16 = np.mean([AVG_As[x][3] for x in range(len(AVG_As))])
A32 = np.mean([AVG_As[x][4] for x in range(len(AVG_As))])


plt.plot([2,4,8,16,32], [A2, A4, A8, A16, A32])
plt.show()


best2 = np.mean([Best_As[x][0] for x in range(len(Best_As))])
best4 = np.mean([Best_As[x][1] for x in range(len(Best_As))])
best8 = np.mean([Best_As[x][2] for x in range(len(Best_As))])
best16 = np.mean([Best_As[x][3] for x in range(len(Best_As))])
best32 = np.mean([Best_As[x][4] for x in range(len(Best_As))])


plt.plot([2,4,8,16,32], [best2, best4, best8, best16, best32])
plt.show()





