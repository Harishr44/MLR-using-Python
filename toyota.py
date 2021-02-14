# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:48:16 2020

@author: Harish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

toyota= pd.read_csv("F://DS//assingmentsMLR//ToyotaCorolla.csv",encoding='latin1')
toyota.drop(toyota.columns[[0,1,4,5,7,9,10,11,14,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]],axis=1,inplace=True)

d1=toyota.corr()
#none of the input variables are corelated so there is no problem of collinearity
# there is some corelation between Quaterly tax and weight= 0.62

toyota.columns 
import statsmodels.formula.api as smf
model1=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit()
model1.params
model1.summary()
# cc and doors p value are insignificant

model2=smf.ols('Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight',data=toyota).fit()
model2.params
model2.summary()
# after eliminating cc and doors we get significant values for all the input variables
pred2=model2.predict(toyota)


import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model2)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse2=sqrt(mean_squared_error(toyota.Price,pred2))
rmse2
#Rs 1339


model3=smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota).fit()
model3.params
model3.summary()
#cc p value is insignificant

model4=smf.ols('Price~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit()
model4.params
model4.summary()
#door value is insignificant


model5=smf.ols('Price~cc+Doors',data=toyota).fit()
model5.params
model5.summary()
# pvalue is significant but rsq value is very low 0.047

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
