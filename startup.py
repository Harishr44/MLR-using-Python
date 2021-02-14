# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:12:05 2020

@author: Harish
"""
#y=profit
#x= r&D Spend, MarketingSpend administration 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
startup= pd.read_csv("50_Startups.csv")

dstate=pd.get_dummies(startup.State)
startup["dstate"]= dstate

startup.corr()
# there is strong corelation between R&D spend and marketing spend (0.72)
# so there is colinearity problem between our independent variables

import seaborn as sns
sns.pairplot(startup)

startup.columns

import statsmodels.formula.api as smf
model1=smf.ols('Profit~RDSpend+Administration+MarketingSpend',data=startup).fit()
model1.params
model1.summary()
# p value adminstration and marketing spend are higher(>0.05) so they are not significant variable to predict

pred1=model1.predict(startup)

#model using only Administration
ml_admin=smf.ols('Profit~Administration',data=startup).fit()
ml_admin.params
ml_admin.summary()
# rsq value=0.04, pvalue=0.162>0.05, both values suggest that administration is not significant variable

#model using Rdspend
ml_rd=smf.ols('Profit~RDSpend',data=startup).fit()
ml_rd.params
ml_rd.summary()
#it is significant rsq=0.947


#model using only marketingspend
ml_market=smf.ols('Profit~MarketingSpend',data=startup).fit()
ml_market.params
ml_market.summary()
#rsq= 0.60, pvalue=0.00<0.05, it is significant variable

#model using admin and marketing spend
model2=smf.ols('Profit~Administration+MarketingSpend',data=startup).fit()
model2.params
model2.summary()
#p value for both the parameter are significant
#we can consider this model for consideration.


#model using rdspend and marketingspend
model3=smf.ols('Profit~RDSpend+MarketingSpend',data=startup).fit()
model3.params
model3.summary()
#Marketingspend pvalue=0.06>0.05 insignificant



import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 49,48,46,45 6,19 showing high influence

startup_new=startup.drop(startup.index[[49,48,46,45,19]],axis=0)
startup_new

#newmodel
mlnew=smf.ols('Profit~RDSpend+Administration+MarketingSpend',data=startup_new).fit()
mlnew.params
mlnew.summary()
#rsq=0.96,pvalue for admin=0.22>0.05 which is insignificant
#after eliminating influncers still we have not reach the optimum p value for model so we have to find out VIF values



#VIF's for rdspend
rsq_rdspend=smf.ols('RDSpend~Administration+MarketingSpend',data=startup_new).fit().rsquared
vif_rdspend=1/(1-rsq_rdspend)
#2.85<10 it is good value


#VIF's for administration
rsq_admin=smf.ols('Administration~RDSpend+MarketingSpend',data=startup_new).fit().rsquared
vif_admin=1/(1-rsq_admin)
#1.196<10 it is also good value

#VIf's formarketingspend
rsq_marketing=smf.ols('MarketingSpend~RDSpend+Administration',data=startup_new).fit().rsquared
vif_marketing=1/(1-rsq_marketing)
#2.78<10


d1={'Variables':['RDSpend','Administration','MarketingSpend'],'VIF':[vif_rdspend,vif_admin,vif_marketing]}
vif_frame= pd.DataFrame(d1)
vif_frame

#eliminating column administration as it has high p value in mlnew
#model using only rdspend and marketing in new dataset
mlnew2=smf.ols('Profit~RDSpend+MarketingSpend',data=startup_new).fit()
mlnew2.params
mlnew2.summary()
#pvalue and rsq value are significant, we can proceed with this model

prednew2=mlnew2.predict(startup_new)
startup_new["prednew2"]=prednew2

sm.graphics.plot_partregress_grid(mlnew2)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(startup_new.Profit,prednew2))
rmse
#6857


#model using Rdspend
ml_rd=smf.ols('Profit~RDSpend',data=startup).fit()
ml_rd.params
ml_rd.summary()
#it is significant rsq=0.947

predrd=ml_rd.predict(startup)
predrd
rmserd=sqrt(mean_squared_error(startup.Profit,predrd))
rmserd
#9226

#As model using just RDSpend has more error than model using new startup dataset in which RDspend and Marketing spend is used
# so we can use mlnew2 model