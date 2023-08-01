# -*- coding: utf-8 -*-

""" BLOCK 1. IMPORT PYTHON MODULES FOR MAIN SECTION OF SCRIPT"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import compress
from scipy.optimize import curve_fit
from time import time
from sklearn.cluster import KMeans

#%%
start=time()
""" BLOCK 2. CREATION OF FUNCTIONS USED IN THIS SCRIPT"""



def boolean_indices(search_array):
    """
    function to return the array indices of clustered data within the orignal (raw data) array

    Parameters
    ----------
        search_array: 2D array_like
        
    Returns
    -------
        Boolean array indices
    """
    return list(compress(range(len(search_array)), search_array))




def rsq(ydata,fit):
    """
    function to calculate the coefficient of determination (r_sq value) between raw data and a fitted model
    from the sum of squares of residuals (ss_res) and the total sum of squares (ss_tot)

    Parameters
    ----------
        ydata: 1D array_like
        fit: 1D array_like
    
    Returns
    -------
        coefficient of determination (r_sq value)
    
    """
    res=ydata-fit
    ss_res=np.sum(res**2)
    ss_tot=np.sum((ydata-np.mean(ydata))**2)
    r_sq=1-(ss_res/ss_tot)
    return r_sq


def Gauss1D(x,*p):
    """
    function to calculate a general one dimensional Gaussian

    Parameters
    ----------
        x: 1D array_like
           coordinate(s) where the function should be evaluated
        p: list
           list of parameters for the Gaussian function
           
    Returns
    -------
        array-like
        the value of the (1D) Gaussian described by the parameters p at
        position (x)
           
    """
    g = p[3] + p[2] * np.exp(-((p[0] - x) / p[1]) ** 2 / 2.)
    return g


def Lorentz1D(x, *p):
    """
    function to calculate a general one dimensional Lorentzian

    Parameters
    ----------
        x: 1D array_like
                coordinate(s) where the function should be evaluated
        p: list
           list of parameters for the Lorentzian function
           
    Returns
    -------
        array-like
        the value of the (1D) Lorentzian described by the parameters p at
        position (x)
    """
    l = p[3] + p[2] / (1 + (2 * (x - p[0]) / p[1]) ** 2)
    return l


def PseudoVoigt1D(x, *p):
    
    """
    function to calculate a general one dimensional Psuedo-Voigt

    Parameters
    ----------
        x:  array-like
            coordinate(s) where the function should be evaluated
        p : list
            list of parameters of the Psuedo-Voigt-function
            [XCEN, FWHM, AMP, BACKGROUND, ETA,M];
            SIGMA = FWHM / (2*sqrt(2*log(2)));
            
    Returns
    -------
        array-like
        the value of the (1D) Psuedo-Voigt described by the parameters p at
        position (x)
    """

    pv = p[4]

    sigma = p[1] / (2 * np.sqrt(2 * np.log(2)))
    f = (x*p[5]+p[3]) + pv * Lorentz1D(x, p[0], p[1], p[2], 0) + \
        (1 - pv) * Gauss1D(x, p[0], sigma, p[2], 0)
    return f

def Gauss2d_P(data_tuple, *p):
    """
    function to calculate a general two dimensional Gaussian

    Parameters
    ----------
        x, y :  array-like
                coordinate(s) where the function should be evaluated
        p :     list
        list of parameters of the Gauss-function
        [AMP,XCEN, YCEN, FWHMX, FWHMY,ANGLE];
        SIGMA = FWHM / (2*sqrt(2*log(2)));
        ANGLE = rotation of the X, Y direction of the Gaussian in radians

    Returns
    -------
        array-like
        the value of the Gaussian described by the parameters p at
        position (x, y)
    """
    (x,y)=data_tuple
    sigma_x,sigma_y=p[3]/(2*np.sqrt(2*np.log(2))),p[4]/(2*np.sqrt(2*np.log(2)))
    rcen_x = p[1] * np.cos(p[5]) - p[2] * np.sin(p[5])
    rcen_y = p[1] * np.sin(p[5]) + p[2] * np.cos(p[5])
    xp = x * np.cos(p[5]) - y * np.sin(p[5])
    yp = x * np.sin(p[5]) + y * np.cos(p[5])

    g = p[0] * np.exp(-(((rcen_x - xp) / sigma_x) ** 2 +
                                  ((rcen_y - yp) / sigma_y) ** 2) / 2.)
    return g.ravel()

def Lorentz2d_P(data_tuple, *p):
    """
    function to calculate a general two dimensional Lorentzian

    Parameters
    ----------
        x, y :  array-like
            coordinate(s) where the function should be evaluated
        p :     list
            list of parameters of the Lorentzian-function
            [AMP,XCEN, YCEN, FWHMX, FWHMY,ANGLE];
            SIGMA = FWHM / (2*sqrt(2*log(2)));
            ANGLE = rotation of the X, Y direction of the Lorentzian in radians

    Returns
    -------
        array-like
            the value of the Lorentzian described by the parameters p at
            position (x, y)
    """
    
    (x,y)=data_tuple
    rcen_x = p[1] * np.cos(p[5]) - p[2] * np.sin(p[5])
    rcen_y = p[1] * np.sin(p[5]) + p[2] * np.cos(p[5])
    xp = x * np.cos(p[5]) - y * np.sin(p[5])
    yp = x * np.sin(p[5]) + y * np.cos(p[5])
    
    l=p[0] / (1+(2*(rcen_x-xp)/p[3])**2 + (2*(rcen_y-yp)/p[4])**2)
    
    return l.ravel()

def PV2D_P(data_tuple,*p):
    """
    function to calculate a general two dimensional Psuedo-Voigt function

    Parameters
    ----------
        x, y :  array-like
            coordinate(s) where the function should be evaluated
        p :     list
                list of parameters of the Psuedo-Voigt-function
                [AMP,XCEN, YCEN, FWHMX, FWHMY,ANGLE];
                SIGMA = FWHM / (2*sqrt(2*log(2)));
                ANGLE = rotation of the X, Y direction of the peak in radians

    Returns
    -------
        array-like
            the value of the Psuedo-Voigt described by the parameters p at
            position (x, y)
    """
    """[AMP, XCEN, YCEN, FWHMX,FWHMY,ANGLE,ETA]"""
    pv=p[6]
    f=pv*Lorentz2d_P(data_tuple,p[0],p[1],p[2],p[3],p[4],p[5]) +\
    (1-pv)*Gauss2d_P(data_tuple,p[0],p[1],p[2],p[3],p[4],p[5])
    
    return f.ravel()



def Plane(data_tuple,*p):
    """
    function to calculate the general form of the equation of a plane
    
    Parameters
    ----------
        x, y :  array-like
            coordinate(s) where the function should be evaluated
    
    
    Returns
    -------
        array-like
            the value of the plane function described by the parameters p at
            position (x, y)
    
    """
    (x,y)=data_tuple
    z = p[0]*x + p[1]*y + p[2]
    return z.ravel()


def PV2D_P_Plane(data_tuple,*p):
    """
    function to calculate a general two dimensional Psuedo-Voigt function with a plane background
    
    Parameters
    ----------
        x, y :  array-like
            coordinate(s) where the function should be evaluated
        p :     list
            list of parameters of the Psuedo-Voigt-function
            SIGMA = FWHM / (2*sqrt(2*log(2)));
            ANGLE = rotation of the X, Y direction of the peak in radians
            ETA = The weighting applied to the constituent 2D Gaussian and Lorentzian functions
            A,B,C - coefficients described in the Plane function.
    
    Returns
    -------
        array-like
            the value of the 2D Psuedo-Voigt with plane background described by the parameters p at
            position (x, y)
    
    
    """
    """ [AMP1,XCEN1,YCEN1,FWHM_X1,FWHM_Y1,ANGLE1,ETA1\
    A,B,C]"""
    p=list(p)
    p1=p[0:7]
    p2=p[7:]
    
    PVP=PV2D_P(data_tuple,*p1)+ Plane(data_tuple,*p2)
    return PVP
    

def TwoPV2D_P_Plane(data_tuple,*p):
    """
    function to calculate two general two dimensional Psuedo-Voigt function with a plane background
    
    Parameters
    ----------
        x, y :  array-like
            coordinate(s) where the function should be evaluated
        p :     list
            list of parameters of the Psuedo-Voigt-function
            SIGMA = FWHM / (2*sqrt(2*log(2))); (for peak 1 (X&Y) or 2 (X&Y))
            ANGLE = rotation of the X, Y direction of the peak in radians
            ETA = The weighting applied to the constituent 2D Gaussian and Lorentzian functions
            A,B,C - coefficients described in the Plane function.
    
    Returns
    -------
        array-like
            the value of the combined 2D Psuedo-Voigt peak (x2) with a plane background described by the parameters p at
            position (x, y)
    
    """
    
    """ [AMP1,XCEN1,YCEN1,FWHM_X1,FWHM_Y1,ANGLE1,ETA1\
        AMP2,XCEN2,YCEN2,FWHM_X2,FWHM_Y2,ANGLE2,ETA2\
        A,B,C]
    """
    
    p=list(p)
    p1=p[0:7]
    p2=p[7:14]
    p3=p[14:]
    
    PV2=PV2D_P(data_tuple,*p1) + PV2D_P(data_tuple,*p2) + Plane(data_tuple,*p3)
    return PV2.ravel()

def threePV2D_P_Plane(data_tuple,*p):
    """
    function to calculate 3 general two dimensional Psuedo-Voigt function combined with a plane background
    
    Parameters
    ----------
        x, y :  array-like
                coordinate(s) where the function should be evaluated
        p :     list
                list of parameters of the Psuedo-Voigt-function
                SIGMA = FWHM / (2*sqrt(2*log(2))); (for peak 1 (X&Y), 2 (X&Y) or 3 (X&Y))
                ANGLE = rotation of the X, Y direction of the peak in radians
                ETA = The weighting applied to the constituent 2D Gaussian and Lorentzian functions
                A,B,C - coefficients described in the Plane function.
    
    Returns
    -------
        array-like
            the value of the 2D Psuedo-Voigt peaks (x3) with a plane background described by the parameters p at
            position (x, y)
    
    """
    
    
    """ [AMP1,XCEN1,YCEN1,FWHM_X1,FWHM_Y1,ANGLE1,ETA1\
        AMP2,XCEN2,YCEN2,FWHM_X2,FWHM_Y2,ANGLE2,ETA2\
        AMP3,XCEN3,YCEN3,FWHM_X3,FWHM_Y3,ANGLE3,ETA3\
        A,B,C]"""

    p=list(p)
    p1=p[0:7]
    p2=p[7:14]
    p3=p[14:21]
    p4=p[21:]
    
    PV3=PV2D_P(data_tuple,*p1) + PV2D_P(data_tuple,*p2) + PV2D_P(data_tuple,*p3) + Plane(data_tuple,*p4)
    return PV3.ravel()


def clstr_fit1(az_rng,q,clstr_n,file_pre,file_sfx,params,fit_array):
    
    """
    function to fit a 1D Psuedo Voigt peak to the 1D XRD data and to the 1D orientation data
    This is only for data where one crystallite population is present.
    Here, two 1D Psuedo-Voigts have been fit to the 1D XRD and orientation data
    to improve stability. Fitting with a 2D Psuedo-Voigt produces poor fits due to noise in
    the azimuth.
    
    Parameters
    ----------
        az_rng: list_like
                lower and upper limits applied to the data to aid with fitting
        q: array-like
           coordinate(s) where the function should be evaluated in q space (for 1D XRD data)
        clustr_n: list_like
                  file indices belonging to a given cluster of data
        file_pre: string
                  file prefix naming convention
        file_sfx: string
                  file suffix naming convention
        params: array_like
                  initial parameters for the Psuedo-Voigt-function
        fit_array: array_like
                   array where fitted data parameters are stored
               
    Returns
    -------
        fit_array: array_like
                   array where fitted data parameters have been stored in the corresponding columns & rows
        
    """
    ### CREATE ARRAY TO STORE INITIAL FITTING PARAMETERS FOR 1D XRD AND ORIENTATION DATA
    pars_arr=np.zeros((12,1))
    ### DEFINE AZIMUTHAL RANGE - FOR INTENSITY ARC 1 
    az_l,az_h=az_rng[0],az_rng[1]
    az_clstr=np.arange(az_l,az_h,1) 
    ### [0:6,0] CORRESPONDS TO PARAMETERS FOR THE 1D XRD DATA (IN Q SPACE)
    ### [6:,0] CORRESPODS TO PARAMETERS FOR THE 1D ORIENTATION DATA (IN AZIMUTHAL COORDINATES)
    pars_arr[0:6,0],pars_arr[6:,0]=params[0:6],params[6:]
    
    for i in clstr_n:

        file_name=file_pre+str(i+1)+file_sfx
        print(file_name)
        corr_data=np.loadtxt(file_name,delimiter=',')[:,1:]
        
        ### FILE NO. IN FIRST COLUMN
        fit_array[i,0]=i+1
        
        ### OBTAIN 1D XRD DATA BY AVERAGING ALONG AXIS 1 
        ### 100 HAS BEEN ADDED TO AVOID ANY NEGATIVE BKG PARAMETERS 
        XRD=np.mean(corr_data,axis=1)+100
        ### SET INITIAL AMPLITUDE PARAMETER TO MAXIMUM VALUE IN ARRAY
        pars_arr[2,0]=np.max(XRD)
        ### FIT PSEUDO-VOIGT TO DATA WITH INITIAL PARAMETERS
        popt, pcov = curve_fit(PseudoVoigt1D, q, XRD, p0=pars_arr[0:6,0])
        ### GENERATE FITTED DATA POINTS
        XRD_data_fitted = PseudoVoigt1D(q, *popt)
    
        ### STORE FITTED PARAMETERS (PEAK POSITION AND FWHM INTO FIT ARRAY)
        fit_array[i,1]=popt[0] ## XRD CEN POS - Q SPACE
        fit_array[i,2]=popt[1] ## XRD FWHM - Q SPACE
        print(rsq(XRD,XRD_data_fitted))
        try:
            ORR=np.mean(corr_data,axis=0)+100
            ORR=ORR[az_l:az_h]
            ### SET INITIAL AMPLITUDE PARAMETER TO MAXIMUM VALUE IN ARRAY
            pars_arr[8,0]=np.max(ORR)
            ### FIT PSEUDO-VOIGT TO DATA WITH INITIAL PARAMETERS
            ORR_popt, ORR_pcov = curve_fit(PseudoVoigt1D, az_clstr, ORR, p0=pars_arr[6:,0])
            ### GENERATE FITTED DATA POINTS
            ORR_data_fitted = PseudoVoigt1D(az_clstr, *ORR_popt)
            print(rsq(ORR,ORR_data_fitted))
            
            ### STORE FITTED PARAMETERS INTO FIT ARRAY
            fit_array[i,3]=ORR_popt[0]  ## ORIENTATION CEN POS - THETA
            fit_array[i,4]=ORR_popt[1]  ## ORIENTATION FWHM - THETA
            fit_array[i,5]=100          ## PK1 % PROPORTION
            fit_array[i,6:]=0           ## PEAK AREA NOT RETURNED HERE AS IT IS USED ONLY FOR CALCULATION OF CRYSTALLITE PROPORTION
            fit_array[i,19]=rsq(ORR,ORR_data_fitted) ## R**2 VALUE
        
        except RuntimeError:
            fit_array[i,:]=0


def clstr_fit2(az_rng,q,clstr_n,file_pre,file_sfx,params,fit_array):
    
    """
    function to fit two general two dimensional Psuedo-Voigt function with a plane background
    to the THETA VS Q CAKE DATA
    
    Parameters
    ----------
        az_rng: list_like
                lower and upper limits applied to the data to aid with fitting
        q: array-like
           coordinate(s) where the function should be evaluated in q space (for 1D XRD data)
        clustr_n: list_like
                  file indices belonging to a given cluster of data
        file_pre: string
                  file prefix naming convention
        file_sfx: string
                  file suffix naming convention
        params: array_like
                  initial parameters for the Psuedo-Voigt-function
                  please see TwoPV2D_P_Plane for the order of parameters
        fit_array: array_like
                   array where fitted data parameters are stored
               
    Returns
    -------
        fit_array: array_like
                   array where fitted data parameters have been stored in the corresponding columns & rows
        
        """
    ### CREATE ARRAY TO STORE INITIAL FITTING PARAMETERS
    pars_arr=np.zeros((17,1))
    pars_arr[1:7,0],pars_arr[8:,0]=params[0:6],params[6:]

    ### DEFINE AZIMUTHAL RANGE - FOR INTENSITY ARC 1
    az_l,az_h=az_rng[0],az_rng[1]
    az_clstr=np.arange(az_l,az_h,1)

    ### GENERATE MESHGRID COORDINATES FOR 2D PEAK FITTING
    x,y=np.meshgrid(az_clstr,q) 
    
    for i in clstr_n:
        try:
            ### READ IN DATA FILES
            clstr_n_fn=file_pre+str(i+1)+file_sfx
            clstr_n_data=(np.loadtxt(clstr_n_fn,delimiter=',')[:,1:]+100)[:,az_l:az_h]
            ###PRINT CURRENT DATA FILE
            print(clstr_n_fn)
        
            #### ADD AMPLITUDE PARAMETER VALUES TO THE ARRAY OF INITIAL PARAMETERS
            pars_arr[0,0],pars_arr[7,0]=np.max(clstr_n_data),np.max(clstr_n_data)
        
            clstr_n_initial_guess=pars_arr
            
            ### FIT TwoPV2D_P_Plane TO DATA WITH INITIAL PARAMETERS
            cn_popt, cn_pcov = curve_fit(TwoPV2D_P_Plane, (x, y), clstr_n_data.ravel(), p0=clstr_n_initial_guess)
            ### GENERATE FITTED DATA POINTS
            c_n_data_fitted = TwoPV2D_P_Plane((x, y), *cn_popt)
            ### CALCULATE R^2 VALUE BETWEEN DATA AND FIT
            print(rsq(clstr_n_data.ravel(),c_n_data_fitted))
        
            ### GENERATE 2D PEAKS WITHOUT PLANE BACKGROUND FOR PEAK INTEGRATION
            data_int_cna=np.reshape(PV2D_P((x, y), cn_popt[0],cn_popt[1],cn_popt[2],cn_popt[3],cn_popt[4],cn_popt[5],cn_popt[6]),(500,az_h-az_l))
            data_int_cnb=np.reshape(PV2D_P((x, y), cn_popt[7],cn_popt[8],cn_popt[9],cn_popt[10],cn_popt[11],cn_popt[12],cn_popt[13]),(500,az_h-az_l))

            #### CALCULATE PEAK AREAS AND PEAK % PROPORTIONS
            pkcA_prop=sum(np.trapz(data_int_cna,dx=1,axis=1))/(sum(np.trapz(data_int_cna,dx=1,axis=1))+sum(np.trapz(data_int_cnb,dx=1,axis=1))) *100
            pkcB_prop=sum(np.trapz(data_int_cnb,dx=1,axis=1))/(sum(np.trapz(data_int_cna,dx=1,axis=1))+sum(np.trapz(data_int_cnb,dx=1,axis=1))) *100
            
            fit_array[i,0]=i+1
            ##### FILL ARRAY WITH PARAMETERS
            fit_array[i,1]=cn_popt[2] ## PK 1 CEN POS - Q SPACE
            fit_array[i,2]=cn_popt[4] ## PK 1 FWHM - Q SPACE
            fit_array[i,3]=cn_popt[1] ## PK 1 CEN POS - THETA
            fit_array[i,4]=cn_popt[3] ## PK 1 FWHM - THETA
            fit_array[i,5]=pkcA_prop ## PK1 % PROPORTION
            fit_array[i,6]=sum(np.trapz(data_int_cna,dx=1,axis=1)) ## PK1 INTEGRATED INTENSITY
            
            fit_array[i,7]=cn_popt[9] ## PK 2 CEN POS - Q SPACE
            fit_array[i,8]=cn_popt[11] ## PK 2 FWHM - Q SPACE
            fit_array[i,9]=cn_popt[8] ## PK 2 CEN POS - THETA
            fit_array[i,10]=cn_popt[10] ## PK 2 FWHM - THETA
            fit_array[i,11]=pkcB_prop  ## PK2 % PROPORTION
            fit_array[i,12]=sum(np.trapz(data_int_cnb,dx=1,axis=1)) ## PK2 INTEGRATED INTENSITY
            
            fit_array[i,19]=rsq(clstr_n_data.ravel(),c_n_data_fitted) ## R^2 VALUE
        
        except RuntimeError:
            print('EXCEPTION RAISED!')
            fit_array[i,:]=0


def clstr_fit3(az_rng,q,clstr_n,file_pre,file_sfx,params,fit_array):
    """
    function to fit three general two dimensional Psuedo-Voigt function with a plane background
    to the THETA VS Q CAKE DATA
    
    Parameters
    ----------
        az_rng: list_like
                lower and upper limits applied to the data to aid with fitting
        q: array-like
           coordinate(s) where the function should be evaluated in q space (for 1D XRD data)
        clustr_n: list_like
                  file indices belonging to a given cluster of data
        file_pre: string
                  file prefix naming convention
        file_sfx: string
                  file suffix naming convention
        params: array_like
                initial parameters for the Psuedo-Voigt-function
                please see threePV2D_P_Plane for the order of parameters
        fit_array: array_like
                   array where fitted data parameters are stored
               
    Returns
    -------
        fit_array: array_like
                   array where fitted data parameters have been stored in the corresponding columns & rows
        
    """
    pars_arr=np.zeros((24,1))
    pars_arr[1:7,0],pars_arr[8:14,0],pars_arr[15:,0]=params[0:6],params[6:12],params[12:]

    ### DEFINE AZIMUTHAL RANGE - FOR INTENSITY ARC 1 
    az_l,az_h=az_rng[0],az_rng[1]
    az_clstr=np.arange(az_l,az_h,1) ### 100<= THETA <= 299

    ### GENERATE MESHGRID COORDINATES FOR 2D PEAK FITTING
    x,y=np.meshgrid(az_clstr,q) 
    
    for i in clstr_n:
        try:
            ### READ IN DATA FILES
            clstr_n_fn=file_pre+str(i+1)+file_sfx
            clstr_n_data=(np.loadtxt(clstr_n_fn,delimiter=',')[:,1:]+100)[:,az_l:az_h]
            ###PRINT CURRENT DATA FILE
            print(clstr_n_fn)
        
            #### ADD AMPLITUDE PARAMETER VALUES TO THE ARRAY OF INITIAL PARAMETERS
            pars_arr[0,0],pars_arr[7,0],pars_arr[14,0]=np.max(clstr_n_data),np.max(clstr_n_data),np.max(clstr_n_data)
        
            clstr_n_initial_guess=pars_arr
            ### FIT threePV2D_P_Plane TO DATA WITH INITIAL PARAMETERS
            cn_popt, cn_pcov = curve_fit(threePV2D_P_Plane, (x, y), clstr_n_data.ravel(), p0=clstr_n_initial_guess)
            ### GENERATE FITTED DATA POINTS
            c_n_data_fitted = threePV2D_P_Plane((x, y), *cn_popt)
            ### CALCULATE R^2 VALUE BETWEEN DATA AND FIT
            print(rsq(clstr_n_data.ravel(),c_n_data_fitted))
    
            ### GENERATE 2D PEAKS WITHOUT PLANE BACKGROUND FOR PEAK INTEGRATION
            data_int_cna=np.reshape(PV2D_P((x, y), cn_popt[0],cn_popt[1],cn_popt[2],cn_popt[3],cn_popt[4],cn_popt[5],cn_popt[6]),(500,az_h-az_l))
            data_int_cnb=np.reshape(PV2D_P((x, y), cn_popt[7],cn_popt[8],cn_popt[9],cn_popt[10],cn_popt[11],cn_popt[12],cn_popt[13]),(500,az_h-az_l))
            data_int_cnc=np.reshape(PV2D_P((x, y), cn_popt[14],cn_popt[15],cn_popt[16],cn_popt[17],cn_popt[18],cn_popt[19],cn_popt[20]),(500,az_h-az_l))

            #### CALCULATE PEAK AREAS AND PEAK % PROPORTIONS
            pkcA_prop=sum(np.trapz(data_int_cna,dx=1,axis=1))/(sum(np.trapz(data_int_cna,dx=1,axis=1))+sum(np.trapz(data_int_cnb,dx=1,axis=1)+sum(np.trapz(data_int_cnc,dx=1,axis=1)))) *100
            pkcB_prop=sum(np.trapz(data_int_cnb,dx=1,axis=1))/(sum(np.trapz(data_int_cna,dx=1,axis=1))+sum(np.trapz(data_int_cnb,dx=1,axis=1)+sum(np.trapz(data_int_cnc,dx=1,axis=1)))) *100
            pkcC_prop=sum(np.trapz(data_int_cnc,dx=1,axis=1))/(sum(np.trapz(data_int_cna,dx=1,axis=1))+sum(np.trapz(data_int_cnb,dx=1,axis=1)+sum(np.trapz(data_int_cnc,dx=1,axis=1)))) *100
            
            fit_array[i,0]=i+1
            
            ##### FILL ARRAY WITH PARAMETERS
            fit_array[i,1]=cn_popt[2] ## PK 1 CEN POS - Q SPACE
            fit_array[i,2]=cn_popt[4] ## PK 1 FWHM - Q SPACE
            fit_array[i,3]=cn_popt[1] ## PK 1 CEN POS - THETA
            fit_array[i,4]=cn_popt[3] ## PK 1 FWHM - THETA
            fit_array[i,5]=pkcA_prop ## PK1 % PROPORTION
            fit_array[i,6]=sum(np.trapz(data_int_cna,dx=1,axis=1)) ## PK1 INTEGRATED INTENSITY
            
            fit_array[i,7]=cn_popt[9] ## PK 2 CEN POS - Q SPACE
            fit_array[i,8]=cn_popt[11] ## PK 2 FWHM - Q SPACE
            fit_array[i,9]=cn_popt[8] ## PK 2 CEN POS - THETA
            fit_array[i,10]=cn_popt[10] ## PK 2 FWHM - THETA
            fit_array[i,11]=pkcB_prop  ## PK2 % PROPORTION
            fit_array[i,12]=sum(np.trapz(data_int_cnb,dx=1,axis=1)) ## PK2 INTEGRATED INTENSITY
            
            fit_array[i,13]=cn_popt[16] ## PK 3 CEN POS - Q SPACE
            fit_array[i,14]=cn_popt[18] ## PK 3 FWHM - Q SPACE
            fit_array[i,15]=cn_popt[15] ## PK 3 CEN POS - THETA
            fit_array[i,16]=cn_popt[17] ## PK 3 FWHM - THETA
            fit_array[i,17]=pkcC_prop   ## PK3 % PROPORTION
            fit_array[i,18]=sum(np.trapz(data_int_cnc,dx=1,axis=1)) ## PK2 INTEGRATED INTENSITY
            
            fit_array[i,19]=rsq(clstr_n_data.ravel(),c_n_data_fitted) ## R**2 VALUE

        except RuntimeError:
            print('EXCEPTION RAISED!')
            fit_array[i,:]=0
#%%
""" BLOCK 3. SET FILE PATH TO XRD CAKE DATA"""
### CHANGE DIRECTORY TO WHERE .DAT FILES ARE STORED
os.chdir('D:\\T_REX_scripts\\PD-ROI2')

#%%
""" BLOCK 4. SET FILE INFORMATION FOR IMPORTING DATA AND FITTING"""

### NUMBER OF FILES IN THE MAP
nF=1066

### DEFINE AZIMUTHAL RANGE FOR INTENSITY ARC (0-360 GIVES THE FULL ANGULAR RANGE) 
az_l,az_up=0,360
az=np.arange(az_l,az_up,1) ### 100<= THETA <= 299


#### LOAD Q COLUMN VALUES
q=np.loadtxt('ipp_313868_5_00000.dat')[:,0]
### GENERATE MESHGRID COORDINATES FOR 2D PEAK FITTING
x,y=np.meshgrid(az,q) 

### DEFINE FILE PRE AND SUFFIX HANDLES FOR USE IN LOOP STRUCTURES
file_pre='ipp_313745_'
file_sfx='_[;3056,;3056]_00000_T.dat'

#%%
"""BLOCK 5. READ IN DATA,  CONDENSE DATA W.R.T AZIMUTH TO GET 1D ORIENTATION DATA. STORE DATA IN THE ORR_ARR
NORMALISE ORR_ARR DATA FOR PCA AND TO EASE FITTING"""

### MAKE A ZERO ARRAY DIMS: NUMBER OF AZIMUTHAL BINS (ROWS) X NUMBER OF PIXELS IN A MAP (COLS)
orr_arr=np.zeros((len(az)-1,nF))

files=np.arange(1,nF+1)

for f in files:
    file_name=file_pre+str(f)+file_sfx
    print(file_name)
    data=np.loadtxt(file_name,delimiter=',')[:,1:]
    orr_1D=np.mean(data,axis=0) ### CONDENSE DATA TO 1D
    orr_arr[:,f-1]=orr_1D/np.max(orr_1D) ### NORMALISE AND STORE IN ARRAY


#%%
""" BLOCK 6. PCA AND K-MEANS"""
pca = PCA(50)  ### SET NO. OF COMPONENTS TO PROJECT TO
projected = pca.fit_transform(np.transpose(orr_arr)) ### COMPUTE COMPONENTS

n_clust=30
### CALCULATE DATA CLUSTERS. NO. OF CLUSTERS = 30
km = KMeans(n_clusters=n_clust, init='random',n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(projected)

plt.figure('K-MEANS CLUSTER PLOT OF PC1 VS. PC2')
### PLOT CLUSTERS ON A PLOT OF PC1 VS. PC2
pca=0
pcb=1
pcc=2

clstr_arr=np.zeros((1066,1))

n_clust_range=np.arange(0,n_clust)
### SET COLOURS FOR PLOTTING SYMBOLS IN PCA PLOT
### RETRIEVE INDICES FOR CLUSTERS OF ORIENTATION DATA
colours=['lightgreen','orange','blue','red','pink','black','purple','yellow','brown','green','darkred','lightblue','white','indigo','darkgreen','darkblue','lightcoral','darkblue','moccasin','darkorange', 'lightyellow','khaki','olive','cyan','crimson','steelblue','gold','lime','grey','darkolivegreen']
for cl in n_clust_range:
    ### PLOT CLUSTERS
    plt.scatter(projected[y_km == cl, pca], projected[y_km == cl, pcb],s=50, c=colours[cl],marker='s', edgecolor='black',label='cluster '+str(cl+1))
    ###OBTAIN INDICES FOR EACH CLUSTER
    clust_name='clstr_'+str(cl+1)
    globals()[clust_name] = boolean_indices(y_km == cl)
    exec('clstr_arr[clstr_'+str(cl+1)+',0]='+str(cl))
plt.scatter(km.cluster_centers_[:, pca], km.cluster_centers_[:, pcb],s=250, marker='*',c='red', edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.grid()
plt.show()

clstr_arr[clstr_6,0],clstr_arr[clstr_8,0],clstr_arr[clstr_11,0],clstr_arr[clstr_14,0],clstr_arr[clstr_24,0],clstr_arr[clstr_29,0]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
plt.figure('DENTICLE CLUSTER PLOT')
plt.imshow(np.reshape(clstr_arr,(41,26)),cmap='rainbow',interpolation='none')
#%%
""" BLOCK 7. MAKE ZERO ARRAY FOR FITTING PARAMETERS"""

### DEFINE NO. PARAMETERS TO GET FROM 2D FITS
nP=18

""" 
LIST OF PARAMETERS IN THE FIT_PAR_ARR ARRAY

[COLUMN 0] FILE NUMBER
[COLUMN 1] PK 1 CEN POS - Q SPACE
[COLUMN 2] PK 1 FWHM - Q SPACE
[COLUMN 3] PK 1 CEN POS - THETA
[COLUMN 4] PK 1 FWHM - THETA
[COLUMN 5] PK1 % PROPORTION
[COLUMN 6] PK1 INTEGRATED INTENSITY

[COLUMN 7] PK 2 CEN POS - Q SPACE
[COLUMN 8] PK 2 FWHM - Q SPACE
[COLUMN 9] PK 2 CEN POS - THETA
[COLUMN 10] PK 2 FWHM - THETA
[COLUMN 11] PK2 % PROPORTION
[COLUMN 12] PK2 INTEGRATED INTENSITY

[COLUMN 13] PK 3 CEN POS - Q SPACE
[COLUMN 14] PK 3 FWHM - Q SPACE
[COLUMN 15] PK 3 CEN POS - THETA
[COLUMN 16] FWHM - THETA
[COLUMN 17] PK3 % PROPORTION
[COLUMN 18] PK3 INTEGRATED INTENSITY

[COLUMN 18] R^2 VALUE

Please note that [COLUMN 0] will always contain a numerical value.
The filling of other columns is dependent on the number of peaks that are fitted.
i.e. if one peak is fit, the remaining columns remain as NaNs.

"""
fit_par_arr=np.zeros((nF,nP+2))### +2 for the file number and the r**2 value
fit_par_arr[:]=np.nan

#%%
""" BLOCK 8. SINGLE PEAK FITTING"""
pars_4=np.asarray((1.805,0.02,200,100,0.5,1,100,70,200,100,0.5,1))
clstr_fit1([0,180],q,clstr_4,file_pre,file_sfx,pars_4,fit_par_arr)

pars_5=np.asarray((1.805,0.02,200,100,0.5,1,290,70,200,100,0.5,1))
clstr_fit1([170,350],q,clstr_5,file_pre,file_sfx,pars_5,fit_par_arr)

pars_7=np.asarray((1.805,0.02,200,100,0.5,1,260,60,200,100,0.5,1))
clstr_fit1([170,350],q,clstr_7,file_pre,file_sfx,pars_7,fit_par_arr)

pars_9=np.asarray((1.805,0.02,200,100,0.5,1,260,70,200,100,0.5,1))
clstr_fit1([170,350],q,clstr_9,file_pre,file_sfx,pars_9,fit_par_arr)

pars_19=np.asarray((1.805,0.02,200,100,0.5,1,100,70,200,100,0.5,1))
clstr_fit1([10,190],q,clstr_19,file_pre,file_sfx,pars_19,fit_par_arr)
#%%
""" BLOCK 9. DOUBLE PEAK FITTING"""
pars_1=np.asarray((155,1.8,30,0.03,0,0.1,185,1.815,25,0.03,0,0.5,0,0,100))
clstr_fit2([70,240],q,clstr_1,file_pre,file_sfx,pars_1,fit_par_arr)

pars_2=np.asarray((210,1.80,25,0.03,0,0.1,235,1.81,25,0.03,0,0.5,0,0,100))
clstr_fit2([130,300],q,clstr_2,file_pre,file_sfx,pars_2,fit_par_arr)

pars_3=np.asarray((53,1.8,30,0.03,0,0.1,150,1.815,30,0.03,0,0.5,0,0,0))
clstr_fit2([0,180],q,clstr_3,file_pre,file_sfx,pars_3,fit_par_arr)

pars_10=np.asarray((100,1.805,60,0.03,0,0.1,140,1.815,40,0.03,0,0.5,0,0,0))
clstr_fit2([10,190],q,clstr_10,file_pre,file_sfx,pars_10,fit_par_arr)

pars_11=np.asarray((220,1.805,45,0.03,0,0.1,290,1.815,45,0.03,0,0.5,0,0,0))
clstr_fit2([160,340],q,clstr_11,file_pre,file_sfx,pars_11,fit_par_arr)

pars_13=np.asarray((200,1.805,60,0.03,0,0.1,235,1.815,45,0.03,0,0.5,0,0,0))
clstr_fit2([120,320],q,clstr_13,file_pre,file_sfx,pars_13,fit_par_arr)

pars_15=np.asarray((100,1.805,30,0.03,0,0.1,140,1.815,40,0.03,0,0.5,0,0,0))
clstr_fit2([10,190],q,clstr_15,file_pre,file_sfx,pars_15,fit_par_arr)

pars_16=np.asarray((80,1.805,30,0.03,0,0.1,120,1.815,30,0.03,0,0.5,0,0,0))
clstr_fit2([10,170],q,clstr_16,file_pre,file_sfx,pars_16,fit_par_arr)

pars_17=np.asarray((100,1.805,25,0.03,0,0.1,135,1.815,25,0.03,0,0.5,0,0,0))
clstr_fit2([30,200],q,clstr_17,file_pre,file_sfx,pars_17,fit_par_arr)

pars_18=np.asarray((60,1.805,30,0.03,0,0.1,135,1.815,35,0.03,0,0.5,0,0,0))
clstr_fit2([10,190],q,clstr_18,file_pre,file_sfx,pars_18,fit_par_arr)

pars_20=np.asarray((95,1.805,40,0.03,0,0.1,175,1.815,40,0.03,0,0.5,0,0,0))
clstr_fit2([40,230],q,clstr_20,file_pre,file_sfx,pars_20,fit_par_arr)

pars_21=np.asarray((108,1.8,30,0.03,0,0.1,140,1.815,25,0.03,0,0.5,0,0,100))
clstr_fit2([30,210],q,clstr_21,file_pre,file_sfx,pars_21,fit_par_arr)

pars_22=np.asarray((70,1.805,50,0.03,0,0.1,135,1.815,30,0.03,0,0.5,0,0,100))
clstr_fit2([10,190],q,clstr_22,file_pre,file_sfx,pars_22,fit_par_arr)

pars_23=np.asarray((100,1.805,70,0.03,0,0.1,145,1.815,35,0.03,0,0.5,0,0,100))
clstr_fit2([50,220],q,clstr_23,file_pre,file_sfx,pars_23,fit_par_arr)

pars_25=np.asarray((45,1.805,40,0.03,0,0.1,120,1.815,40,0.03,0,0.5,0,0,100))
clstr_fit2([0,170],q,clstr_25,file_pre,file_sfx,pars_25,fit_par_arr)

pars_26=np.asarray((80,1.805,45,0.03,0,0.1,150,1.815,45,0.03,0,0.5,0,0,100))
clstr_fit2([0,170],q,clstr_26,file_pre,file_sfx,pars_26,fit_par_arr)

pars_27=np.asarray((45,1.805,55,0.03,0,0.1,70,1.815,70,0.03,0,0.5,0,0,100))
clstr_fit2([0,160],q,clstr_27,file_pre,file_sfx,pars_27,fit_par_arr)

pars_28=np.asarray((140,1.805,40,0.03,0,0.1,150,1.815,25,0.03,0,0.5,0,0,100))
clstr_fit2([70,220],q,clstr_28,file_pre,file_sfx,pars_28,fit_par_arr)

pars_29=np.asarray((70,1.805,35,0.03,0,0.1,160,1.815,30,0.03,0,0.5,0,0,100))
clstr_fit2([20,210],q,clstr_29,file_pre,file_sfx,pars_29,fit_par_arr)

pars_30=np.asarray((225,1.805,50,0.03,0,0.1,290,1.815,50,0.03,0,0.5,0,0,100))
clstr_fit2([140,340],q,clstr_30,file_pre,file_sfx,pars_30,fit_par_arr)
#%%
""" BLOCK 10. TRIPLE PEAK FITTING"""
pars_12=np.asarray((120,1.805,50,0.03,0,0.1,165,1.81,40,0.03,0,0.5,190,1.815,30,0.03,0,0.5,0,0,100))
clstr_fit3([60,240],q,clstr_12,file_pre,file_sfx,pars_12,fit_par_arr)
###############################################################################
###############################################################################
###############################################################################
end=time()
print('Run time (s):', end-start)
#%%
""" BLOCK 11. OUTPUT ARRAY AS TEXT FILE"""

np.savetxt('PD_ROI2_2D_FIT_DATA.csv',fit_par_arr,delimiter=',')
