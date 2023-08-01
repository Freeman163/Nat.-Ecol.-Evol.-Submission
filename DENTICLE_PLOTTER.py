# -*- coding: utf-8 -*-

""" BLOCK 1. IMPORT PYTHON MODULES FOR MAIN SECTION OF SCRIPT"""

import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
#%%
""" BLOCK 2. CREATION OF FUNCTIONS USED IN THIS SCRIPT"""

def q_to_d(q):
    
    """
    function to convert values in q space to real space (d) values

    Parameters
    ----------
        q: array_like
        
    Returns
    -------
        Boolean array indices
    """
    
    return ((2*np.pi)/q)


def orr_plot(xstep,ystep,th_data,dims,r=0.8):
    """
    function to visualise a stick plot of crystallite orientation 
    
    Parameters
    ----------
        xstep: scalar
               The x-direction step size (in pixels) between individually plotted sticks
        ystep: scalar
               The y-direction step size (in pixels) between individually plotted sticks
        th_data: array_like
                Crystallite population orientation direction (in degrees)
                This was obtained from the peak fitting of the orientation data
                in CLUSTER_2DFIT.py
        dims: scalar
              values for the shape (rows,columns) of the map to be plotted
        r: scalar, optional
           Length of the individual stick (as a fraction of a unit pixel length)
    
    Returns
    -------
        A stick plot of the orientations of the constituent crystallite populations
    
    """
    [X,Y]=np.meshgrid(np.arange(0,np.shape(th_data)[1]*xstep,xstep),np.arange(0,np.shape(th_data)[0]*ystep,ystep))
    rad_plt_L=np.radians(th_data)
    X_off,Y_off=0.05,0.0
    a_len=((xstep**2)+(ystep**2))**0.5
    
    U=(X) + (a_len*r)*np.cos((-1*np.reshape(rad_plt_L,(dims[0],dims[1]))))
    V=(Y) + (a_len*r)*np.sin((-1*np.reshape(rad_plt_L,(dims[0],dims[1]))))
    
    plt.quiver(X+X_off, Y+Y_off, (U-X),(V-Y),scale=1, units='xy',color='black',headwidth=1,headlength=0.1,pivot='middle',width=0.05)

#%%
""" BLOCK 3. SET FILE PATH TO FIT PARAMETERS DATA"""
start=time()
os.chdir('D:\\T_REX_scripts')
data=np.loadtxt('PD_ROI2_2D_FIT_DATA.csv',delimiter=',')

### DEFINE SHAPE OF MAP (rows,columns)
rr,cc=41,26 


#%%
""" BLOCK 4. OVERLAY ORIENTATION STICK PLOT ON CRYSTALLOGRAPHIC C LATTICE PARAMETER FOR EACH CRYSTALLITE POPULATION"""


###POPULATION 1
pk_th=np.reshape(data[:,3],(rr,cc))

###PLOT THE CRYSTALLOGRAPHIC C LATTICE PARAMETER
plt.figure("Population 1 - Crystallographic c lattice parameter with preferred orientation overlaid")
plt.imshow(np.reshape(2*q_to_d(data[:,1]),(rr,cc)),clim=(6.91,6.97),cmap='jet',interpolation='none')
plt.title("Population 1 - Crystallographic c lattice parameter with preferred orientation overlaid")
plt.colorbar(orientation='vertical',label='c lattice parameter (Å) - crystallite population 1')
POP1=pk_th.copy()

###HERE, THE ORIENTATION STICK PLOTS HAVE BEEN REMOVED OVER THE DENTINE REGION
POP1[0,:-9],POP1[1,:-8],POP1[2,:-7],POP1[3,:-7],POP1[4,:-7],POP1[5,:-7]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[6,:-8],POP1[7,:-9],POP1[8,:-11],POP1[9,:-13],POP1[10,:-15],POP1[11,:-18]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[12,:-20],POP1[13,:-20],POP1[14,:-20],POP1[15,:-19],POP1[16,:-18],POP1[17,:-16]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[18,:-15],POP1[19,:-13],POP1[20,:-11],POP1[21,:-11],POP1[22,:-10],POP1[23,:-9],POP1[24,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[25,:-9],POP1[26,:-9],POP1[27,:-9],POP1[28,:-9],POP1[29,:-9],POP1[30,:-9],POP1[31,:-10]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[32,:-12],POP1[33,:-14],POP1[34,:-17],POP1[35,:-19],POP1[36,:-21],POP1[37,:-22],POP1[38,:-22]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[39,:-22],POP1[40,:-21]=np.nan,np.nan

###OVERLAY THE ORIENTATION DATA AS A STICK PLOT
orr_plot(1,1,POP1,[rr,cc]) 


###POPULATION 2
###PLOT THE CRYSTALLOGRAPHIC C LATTICE PARAMETER
plt.figure("Population 2 - Crystallographic c lattice parameter with preferred orientation overlaid")
plt.imshow(np.reshape(2*q_to_d(data[:,7]),(rr,cc)),clim=(6.91,6.97),cmap='jet',interpolation='none')
plt.title("Population 2 - Crystallographic c lattice parameter with preferred orientation overlaid")
plt.colorbar(orientation='vertical',label='c lattice parameter (Å) - crystallite population 2')
idx=np.where(data[:,9] == 0)[0]
data[idx,9]=np.nan
POP2=np.reshape(data[:,9],(rr,cc)).copy()

###HERE, THE ORIENTATION STICK PLOTS HAVE BEEN REMOVED OVER THE DENTINE REGION
POP2[0,:-9],POP2[1,:-8],POP2[2,:-7],POP2[3,:-7],POP2[4,:-7],POP2[5,:-7],POP2[6,:-8]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[7,:-9],POP2[8,:-11],POP2[9,:-13],POP2[10,:-15],POP2[11,:-18],POP2[12,:-20],POP2[13,:-20]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[14,:-20],POP2[15,:-19],POP2[16,:-18],POP2[17,:-16],POP2[18,:-15],POP2[19,:-13],POP2[20,:-11]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[21,:-11],POP2[22,:-10],POP2[23,:-9],POP2[24,:-9],POP2[25,:-9],POP2[26,:-9],POP2[27,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[28,:-9],POP2[29,:-9],POP2[30,:-9],POP2[31,:-10],POP2[32,:-12],POP2[33,:-14],POP2[34,:-17]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[35,:-19],POP2[36,:-21],POP2[37,:-22],POP2[38,:-22],POP2[39,:-22],POP2[40,:-21]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

###OVERLAY THE ORIENTATION DATA AS A STICK PLOT
orr_plot(1,1,POP2,[rr,cc]) ### ORR PLOT FOR POP 2


###POPULATION 3
###PLOT THE CRYSTALLOGRAPHIC C LATTICE PARAMETER
plt.figure("Population 3 - Crystallographic c lattice parameter with preferred orientation overlaid")
plt.imshow(np.reshape(2*q_to_d(data[:,13]),(rr,cc)),clim=(6.91,6.97),cmap='jet',interpolation='none')
plt.title("Population 3 - Crystallographic c lattice parameter with preferred orientation overlaid")
plt.colorbar(orientation='vertical',label='c lattice parameter (Å) - crystallite population 3')
idx=np.where(data[:,15] == 0)[0]
data[idx,15]=np.nan
POP3=np.reshape(data[:,15],(rr,cc)).copy()

###HERE, THE ORIENTATION STICK PLOTS HAVE BEEN REMOVED OVER THE DENTINE REGION
POP3[0,:-9],POP3[1,:-8],POP3[2,:-7],POP3[3,:-7],POP3[4,:-7],POP3[5,:-7],POP3[6,:-8],POP3[7,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[8,:-11],POP3[9,:-13],POP3[10,:-15],POP3[11,:-18],POP3[12,:-20],POP3[13,:-20],POP3[14,:-20],POP3[15,:-19]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[16,:-18],POP3[17,:-16],POP3[18,:-15],POP3[19,:-13],POP3[20,:-11],POP3[21,:-11],POP3[22,:-10],POP3[23,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[24,:-9],POP3[25,:-9],POP3[26,:-9],POP3[27,:-9],POP3[28,:-9],POP3[29,:-9],POP3[30,:-9],POP3[31,:-10]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[32,:-12],POP3[33,:-14],POP3[34,:-17],POP3[35,:-19],POP3[36,:-21],POP3[37,:-22],POP3[38,:-22],POP3[39,:-22],POP3[40,:-21]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

###OVERLAY THE ORIENTATION DATA AS A STICK PLOT
orr_plot(1,1,POP3,[rr,cc]) ### ORR PLOT FOR POP 3
#%%
""" BLOCK 5. PLOT ALL PREFERRED ORIENTATIONS IN THE ENAMEL ONLY"""
blk_arr=np.zeros((rr,cc))
blk_arr[:,:]=np.nan
plt.figure("Preferred Orientation Stick Plot")
plt.imshow(blk_arr)
orr_plot(1,1,POP1,[rr,cc])
orr_plot(1,1,POP2,[rr,cc])
orr_plot(1,1,POP3,[rr,cc])

#%%
""" BLOCK 6. ORIENTATION PLOTS FOR EACH CRYSTALLITE POPULATION OVERLAID ON TEXTURE (FWHM)"""
###POPULATION 1
plt.figure("Population 1 - Texture (FWHM) with preferred orientation overlaid")
FWHM_1=np.reshape(data[:,4],(rr,cc))
FWHM_1[2,19],FWHM_1[18,13],FWHM_1[19,12],FWHM_1[20,11],FWHM_1[34,10],FWHM_1[39,9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
plt.imshow(FWHM_1,clim=(20,120),cmap='jet_r')
plt.colorbar()

###HERE, THE ORIENTATION STICK PLOTS HAVE BEEN REMOVED OVER THE DENTINE REGION
POP1[0,:-9],POP1[1,:-8],POP1[2,:-7],POP1[3,:-7],POP1[4,:-7],POP1[5,:-7]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[6,:-8],POP1[7,:-9],POP1[8,:-11],POP1[9,:-13],POP1[10,:-15],POP1[11,:-18]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[12,:-20],POP1[13,:-20],POP1[14,:-20],POP1[15,:-19],POP1[16,:-18],POP1[17,:-16]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[18,:-15],POP1[19,:-13],POP1[20,:-11],POP1[21,:-11],POP1[22,:-10],POP1[23,:-9],POP1[24,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[25,:-9],POP1[26,:-9],POP1[27,:-9],POP1[28,:-9],POP1[29,:-9],POP1[30,:-9],POP1[31,:-10]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[32,:-12],POP1[33,:-14],POP1[34,:-17],POP1[35,:-19],POP1[36,:-21],POP1[37,:-22],POP1[38,:-22]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP1[39,:-22],POP1[40,:-21]=np.nan,np.nan

orr_plot(1,1,POP1,[rr,cc]) ### ORR PLOT FOR POP 1

###POPULATION 2
plt.figure("Population 2 - Texture (FWHM) with preferred orientation overlaid")
idx=np.where(data[:,10] == 0)[0]
data[idx,10]=np.nan
FWHM_2=np.reshape(data[:,10],(rr,cc))
FWHM_2[34,21]=np.nan
FWHM_2[24,19]=np.nan
FWHM_2[24,16]=np.nan
plt.imshow(FWHM_2,clim=(20,120),cmap='jet_r')
plt.colorbar()

###HERE, THE ORIENTATION STICK PLOTS HAVE BEEN REMOVED OVER THE DENTINE REGION
POP2[0,:-9],POP2[1,:-8],POP2[2,:-7],POP2[3,:-7],POP2[4,:-7],POP2[5,:-7],POP2[6,:-8]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[7,:-9],POP2[8,:-11],POP2[9,:-13],POP2[10,:-15],POP2[11,:-18],POP2[12,:-20],POP2[13,:-20]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[14,:-20],POP2[15,:-19],POP2[16,:-18],POP2[17,:-16],POP2[18,:-15],POP2[19,:-13],POP2[20,:-11]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[21,:-11],POP2[22,:-10],POP2[23,:-9],POP2[24,:-9],POP2[25,:-9],POP2[26,:-9],POP2[27,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[28,:-9],POP2[29,:-9],POP2[30,:-9],POP2[31,:-10],POP2[32,:-12],POP2[33,:-14],POP2[34,:-17]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP2[35,:-19],POP2[36,:-21],POP2[37,:-22],POP2[38,:-22],POP2[39,:-22],POP2[40,:-21]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
orr_plot(1,1,POP2,[rr,cc]) ### ORR PLOT FOR POP 2

###POPULATION 3
plt.figure("Population 3 - Texture (FWHM) with preferred orientation overlaid")
idx=np.where(data[:,16] == 0)[0]
data[idx,16]=np.nan
plt.imshow(np.reshape(data[:,16],(rr,cc)),clim=(20,120),cmap='jet_r')
plt.colorbar()

###HERE, THE ORIENTATION STICK PLOTS HAVE BEEN REMOVED OVER THE DENTINE REGION
POP3[0,:-9],POP3[1,:-8],POP3[2,:-7],POP3[3,:-7],POP3[4,:-7],POP3[5,:-7],POP3[6,:-8],POP3[7,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[8,:-11],POP3[9,:-13],POP3[10,:-15],POP3[11,:-18],POP3[12,:-20],POP3[13,:-20],POP3[14,:-20],POP3[15,:-19]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[16,:-18],POP3[17,:-16],POP3[18,:-15],POP3[19,:-13],POP3[20,:-11],POP3[21,:-11],POP3[22,:-10],POP3[23,:-9]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[24,:-9],POP3[25,:-9],POP3[26,:-9],POP3[27,:-9],POP3[28,:-9],POP3[29,:-9],POP3[30,:-9],POP3[31,:-10]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
POP3[32,:-12],POP3[33,:-14],POP3[34,:-17],POP3[35,:-19],POP3[36,:-21],POP3[37,:-22],POP3[38,:-22],POP3[39,:-22],POP3[40,:-21]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

orr_plot(1,1,POP3,[rr,cc]) ### ORR PLOT FOR POP 3

end=time()

print('Run time (s):', end-start)
