# Nat.-Ecol.-Evol.-Submission
Scripts and data associated with the submission of the manuscript "Iron-coated Komodo dragon teeth and the complex dental enamel of carnivorous reptiles"

This readme file describes Python code associated with the manuscript 'Iron-coated Komodo dragon teeth and the complex dental enamel of carnivorous reptiles'.

**FILE OVERVIEW – THESE FILES ARE FOR DEMO PURPOSES**

**File:** CLUSTER_2DFIT.py

**Description:** Python file to conduct PCA and K-means clustering on the ‘caked’ X-ray diffraction data and fit 2D Pseudo-Voigt peaks with a plane background on to the (002) diffraction peak(s). The ‘caked’ data was exported using the DAWN software package (https://dawnsci.org/). Fitting parameters are stored within an array and outputted for visualising data in DENTICLE_PLOTTER.py. Here, the script is set to read data corresponding to sample UALVP 53472 - region PD-ROI2.

**File:** DENTICLE_PLOTTER.py

**Description:** A file to visualise the crystallographic c lattice parameters, texture and preferred orientation (obtained from intensity arcs in the (002) diffraction peaks) for the constituent crystallite populations obtained from the fitting of the ‘caked’ X-ray diffraction data performed by CLUSTER_2DFIT.py.

**File(s):**.dat demo files

**Description:** Example files of pre-processed ‘caked’ X-ray diffraction data about the (002) diffraction peak(s) for running cluster analysis and 2D peak fitting with the CLUSTER_2DFIT.py file. The files have the following naming convention;

ipp_313745_i_[;3056,;3056]_00000_T 

where i refers to the file number.

Please also find the file PD_ROI2_2D_FIT_DATA.csv, which is the output array from CLUSTER_2DFIT.py containing the fitting parameters to be used for data visualisation in DENTICLE_PLOTTER.py

**SOFTWARE DEPENDENCIES:**

To run the scripts, Python needs to be installed. These scripts have been composed and tested on Python 3.9.12, via the Anaconda installation. Installation instructions for the Anaconda distribution of Python can be found here: (https://www.anaconda.com/download).

The Python scripts were run using the following modules;

Numpy 1.21.5
Matplotlib 3.5.1
Scipy 1.7.3
Sklearn 1.0.2

The Python scripts have been tested on Windows 10.

**EXPECTED INSTALLATION TIME:** 

The Anaconda installation of Python takes approximately 15-20 minutes typically to install, but this will vary according to the specifications of the computer.

**EXPECTED SCRIPT RUN TIME:**

Run times for the respective .py files are reported using a laptop with; 
Processor:	11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, 1690 Mhz, 4 Core(s), 8 Logical Processor(s)
Installed Physical Memory (RAM):	16.0 GB

CLUSTER_2DFIT.py: ~ 1.5 hrs
DENTICLE_PLOTTER.py: ~ 0.2-0.3 s

The Python files within this folder can be ran either by opening the file in the Anaconda distribution of Python and clicking the ‘run’ button or directly from the command line using 'python3.9 filename.py'.

**DEMO INSTRUCTIONS:**

After installing the relevant software:
1.	Download the ‘caked’ X-ray diffraction .dat files.
2.	Open the file CLUSTER_2DFIT.py in a text editor and change the directory path in BLOCK 3 to where the .dat files have been stored.
3.	Run CLUSTER_2DFIT.py.
4.	Download the file PD_ROI2_2D_FIT_DATA.csv.
5.	Open the file DENTICLE_PLOTTER.py in an text editor and change the directory path in BLOCK 3 to where the .dat files have been stored.
6.	Run DENTICLE_PLOTTER.py.
