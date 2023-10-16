import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
plt.style.use("default")

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'errorbar.capsize': 2})

# Avoid Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import cycler
COEFF = 1.048576

n = 4 #number of lines
color = plt.cm.Oranges(np.linspace(0.3, 1,n)) #gnuplot - colormap name, 0 and 1 determine the boundaries of the color
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

def _get_yerr(df):
    alpha=0.05
    std = df.groupby('count').std()
    n = df.groupby('count').count()
    #Calculate confidence intervals 
    yerr = std / np.sqrt(n) * st.t.ppf(1-alpha/2, n - 1)
    yerr = yerr.fillna(0)
    return yerr

def plot(scenario):
 #   if scenario == 'basic':
        #BEB = [38.36, 33.95, 30.58, 27.03]
        #BEB = [38.36, 33.95, 30.58, 27.03]
        #RANGE = [5, 15, 30, 50]

    #elif scenario == 'convergence':
     #   BEB = [38.04, 35.90, 33.41, 31.21]
     #   RANGE = [6, 15, 30, 50]

    df = pd.read_csv("my-final-descentralized_sta_tput.csv")
    #df2 = pd.read_csv("my-final-queue-level.csv")
    #df = pd.read_csv("orig_final.csv")
    #BEB_df = pd.read_csv("my_static.csv")
    #for i, nwifi in enumerate(RANGE):
     #   filt = BEB_df["count"]==nwifi

        #weird = BEB_df[filt].values[0][-1]
        #coeff = BEB[i]/weird

        #BEB_df.loc[np.where([filt])[1], "Thr"] = BEB_df[filt]["Thr"]

    df = df[df['type']==scenario]
    #df2 = df2[df2['type']==scenario]
    
    DDPG = df[df['algorithm']=='ddpg'].groupby('count').mean()*COEFF
    DQN = df[df['algorithm']=='dqn'].groupby('count').mean()*COEFF
    
    #DDPG2 = df2[df2['algorithm']=='ddpg'].groupby('count').mean()*COEFF
    #DQN2 = df2[df2['algorithm']=='dqn'].groupby('count').mean()*COEFF
    #STATIC = df[df['algorithm']=='static'].groupby('count').mean()*COEFF
    #BEB_m = BEB_df.groupby(["count", "RngRun"]).sum().groupby('count').mean()
    #print(BEB_m)

    #DDPG_yerr = _get_yerr(df[df['algorithm']=='ddpg'])
    #BEB_yerr = _get_yerr(BEB_df.groupby(["count", "RngRun"]).sum())
    #DQN_yerr = _get_yerr(df[df['algorithm']=='dqn'])
    
    #DDPG_yerr2 = _get_yerr(df2[df2['algorithm']=='ddpg'])
    #DQN_yerr2 = _get_yerr(df2[df2['algorithm']=='dqn'])

    plt.figure(figsize=(6.4, 4.8),dpi=100)

    #plt.errorbar(RANGE, BEB, fmt='.-', label="BEB", marker="s", markersize=6, yerr=[0, 0, 0, 0])
    #.errorbar(RANGE, STATIC.values, fmt='.-', label="Look-up table", markersize=10, yerr=[0, 0, 0, 0])
    DQN.plot(label="Decentralized-DQN", marker="v",markersize=6, ax=plt.gca(), color="green")
    
    DDPG.plot(label='Decentralized-DDPG', marker="^",markersize=6, ax=plt.gca(), color="deepskyblue")
    
    #DQN2.plot(fmt='.-', label="DQN-Avg Queue Level", marker="v",markersize=6, yerr=DQN_yerr2, ax=plt.gca(), color="green")
        
    #DDPG2.plot(fmt='.-', label='DDPG-Avg Queue Level', marker="^",markersize=6, yerr=DDPG_yerr2, ax=plt.gca(), color="deepskyblue")
   

    plt.xlabel("Number of stations")
    if (scenario=='convergence'):
        print("Fig. 6. Network Throughput for the Dynamic Topology")
        plt.xlabel("Number of stations")
    plt.ylabel("Network throughput [Mb/s]")
    #plt.ylim([26, 42])
    plt.xlim([0, 55])
    #plt.title("CONVERGENCE scenario comparison")
    #plt.legend(["BEB", "DQN", "DDPG"], loc=3, frameon=False)
    #plt.legend(["BEB", "DQN-Avg Queue Level", "DQN-Pcol", "DDPG-Avg Queue Level", "DDPG-Pcol"], loc=3, frameon=False)
    plt.legend(["Decentralized-DQN", "Decentralized-DDPG"], loc=3, fontsize=12, frameon=False)
    
    if (scenario=='basic'):
        print("Fig. 2. Network Throughput for the Static Topology")
        #plt.arrow(40,30,0,8,width=0.1,head_width=0.5,fill=False,color='grey')
        #plt.annotate("Improvement\nover BEB", (41,34),color='grey',fontsize=12)
    #else:
        #plt.arrow(40,34.5,0,3,width=0.1,head_width=0.5,fill=False,color='grey')
        #plt.annotate("Improvement\nover BEB", (41,36),color='grey',fontsize=12)        
    plt.tight_layout()
    #plt.savefig(scenario+'.pdf');
    plt.show()

plot('basic')
plot('convergence')
