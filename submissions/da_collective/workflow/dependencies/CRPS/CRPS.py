#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 09:19:41 2021

@author: Naveen GOUTHAM
"""
import numpy as np

class CRPS:
    '''
    A class to compute the continuous ranked probability score (crps) (Matheson and Winkler, 1976; Hersbach, 2000), the fair-crps (fcrps) (Ferro et al., 2008), and the adjusted-crps (acrps) (Ferro et al., 2008) given an ensemble prediction and an observation.
        
    The CRPS is a negatively oriented score that is used to compare the empirical distribution of an ensemble prediction to a scalar observation.
    
    References:
        [1] Matheson, J. E. & Winkler, R. L. Scoring Rules for Continuous Probability Distributions. Management Science 22, 1087–1096 (1976).
        [2] Hersbach, H. Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems. Wea. Forecasting 15, 559–570 (2000).
        [3] Ferro, C. A. T., Richardson, D. S. & Weigel, A. P. On the effect of ensemble size on the discrete and continuous ranked probability scores. Meteorological Applications 15, 19–24 (2008).

    ----------
    
    Parameters:
        ensemble_members: numpy.ndarray
            The predicted ensemble members. They will be sorted automatically.
            Ex: np.array([2.1,3.5,4.7,1.2,1.3,5.2,5.3,4.2,3.1,1.7])
            
        observation: float
            The observed value.
            Ex: 5.4
            
        adjusted_ensemble_size: int, optional
            The size the ensemble needs to be adjusted to before computing the Adjusted Continuous Ranked Probability Score.
            The default is 200. 
            Note: The crps becomes equal to acrps when adjusted_ensemble_size equals the length of the ensemble_members.
    
    ----------
    
    Method(s):
            
        compute():
            Computes the continuous ranked probability score (crps), the fair-crps (fcrps), and the adjusted-crps (acrps).

    ----------
            
    Attributes:
        crps: Continuous Ranked Probability Score
            It is the integral of the squared difference between the CDF of the forecast ensemble and the observation.
            
            .. math:: 
            \mathrm{crps = \int_{-\infty}^{\infty} [F(y) - F_{o}(y)]^2 dy}
          
        fcrps: Fair-Continuous Ranked Probability Score
            It is the crps computed assuming an infinite ensemble size.
            
            .. math::
            \mathrm{fcrps = crps - \int_{-\infty}^{\infty} [F(y) (1 - F(y))/(m-1)] dy},
            
            where m is the current ensemble size (here: length of ensemble_members)

        acrps: Adjusted-Continuous Ranked Probability Score
            It is the crps computed assuming an ensemble size of M.
            
            .. math::
            \mathrm{acrps = crps - \int_{-\infty}^{\infty} [(1 - m/M) F(y) (1 - F(y))/(m-1)] dy},
            
            where M is the adjusted_ensemble_size
            
    ----------
    
    Demonstration:
    
    import numpy as np
	import CRPS.CRPS as pscore

	Example - 1:
	In [1]: pscore(np.arange(1,5),3.5).compute()
	Out[1]: (0.625, 0.4166666666666667, 0.42083333333333334)

	Example - 2:
	In [2]: crps,fcrps,acrps = pscore(np.arange(1,11),8.3,50).compute()
	In [3]: crps
	Out[3]: 1.6300000000000003
	In [4]: fcrps
	Out[4]: 1.446666666666667
	In [5]: acrps
	Out[5]: 1.4833333333333336
        
    '''
    def __init__(self,ensemble_members,observation,adjusted_ensemble_size=200):
        '''
        Parameters:
            ensemble_members: numpy.ndarray
                The predicted ensemble members.
                Ex: np.array([2.1,3.5,4.7,1.2,1.3,5.2,5.3,4.2,3.1,1.7])
            observation: float
                The observed value.
                Ex: 5.4
            adjusted_ensemble_size: int, optional
                The size the ensemble needs to be adjusted to before computing the Adjusted Continuous Ranked Probability Score.
                The default is 200. 
                Note: The crps becomes equal to acrps when adjusted_ensemble_size equals the length of the ensemble_members.
        
        Returns
        -------
        None
            
        '''
        self.fc = np.sort(ensemble_members)
        self.ob = observation
        self.__m = len(self.fc)
        self.M = int(adjusted_ensemble_size)
        self.__cdf_fc = None
        self.__cdf_ob = None
        self.__delta_fc = None
        self.crps = None
        self.fcrps = None
        self.acrps = None
        
    def __str__(self):
        "Kindly refer to the __doc__ method for documentation. i.e. print(CRPS.__doc__)."
            
    def compute(self):
        '''
        Returns
        -------
        crps, fair-crps, adjusted-crps
        
        '''
        if (self.ob is not np.nan) and (not np.isnan(self.fc).any()):
            if self.ob < self.fc[0]:
                self.__cdf_fc = np.linspace(0,(self.__m - 1)/self.__m,self.__m)
                self.__cdf_ob = np.ones(self.__m)
                all_mem = np.array([self.ob] + list(self.fc), dtype = object)
                self.__delta_fc = np.array([all_mem[n+1] - all_mem[n] for n in range(len(all_mem)-1)], dtype=object)
                
            elif self.ob > self.fc[-1]:
                self.__cdf_fc = np.linspace(1/self.__m,1,self.__m)
                self.__cdf_ob = np.zeros(self.__m)
                all_mem = np.array(list(self.fc) + [self.ob], dtype = object)
                self.__delta_fc = np.array([all_mem[n+1] - all_mem[n] for n in range(len(all_mem)-1)], dtype=object) 
    
            elif self.ob in self.fc:
                self.__cdf_fc = np.linspace(1/self.__m,1,self.__m)
                self.__cdf_ob = (self.fc >= self.ob)
                all_mem = self.fc
                self.__delta_fc = np.array([all_mem[n+1] - all_mem[n] for n in range(len(all_mem)-1)] + list(np.zeros(1)), dtype=object) 
    
            else:
                cdf_fc = []
                cdf_ob = []
                delta_fc = []
                for f in range(len(self.fc)-1):
                    if (self.fc[f] < self.ob) and (self.fc[f+1] < self.ob):
                        cdf_fc.append((f+1)*1/self.__m)
                        cdf_ob.append(0)
                        delta_fc.append(self.fc[f+1] - self.fc[f])
                    elif (self.fc[f] < self.ob) and (self.fc[f+1] > self.ob):
                        cdf_fc.append((f+1)*1/self.__m)
                        cdf_fc.append((f+1)*1/self.__m)
                        cdf_ob.append(0)
                        cdf_ob.append(1)
                        delta_fc.append(self.ob - self.fc[f])
                        delta_fc.append(self.fc[f+1] - self.ob)
                    else:
                        cdf_fc.append((f+1)*1/self.__m)
                        cdf_ob.append(1)
                        delta_fc.append(self.fc[f+1] - self.fc[f])
                self.__cdf_fc = np.array(cdf_fc)
                self.__cdf_ob = np.array(cdf_ob)
                self.__delta_fc = np.array(delta_fc)
            
            self.crps = np.sum(np.array((self.__cdf_fc - self.__cdf_ob) ** 2)*self.__delta_fc)
            if self.__m == 1:
                self.fcrps = self.acrps = 'Not defined'
            else:
                self.fcrps = self.crps - np.sum(np.array(((self.__cdf_fc * (1 - self.__cdf_fc))/(self.__m-1))*self.__delta_fc))
                self.acrps = self.crps - np.sum(np.array((((1 - (self.__m/self.M)) * self.__cdf_fc * (1 - self.__cdf_fc))/(self.__m-1))*self.__delta_fc))
            return self.crps, self.fcrps, self.acrps
        else:
            return np.nan, np.nan, np.nan
        
        