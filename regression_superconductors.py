'''Superconductors Data Analysis
Develop by: Nick Belgau'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import norm

#https://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance#
file = 'superconductors.csv'
targetvar = 'mean_ThermalConductivity'

df = pd.read_csv(file)
ds = df

print('Initial variable list:')
for i,col in enumerate(df.columns): #Show number of columns and names
    print(i,col)
 
'''Feature Selection'''
#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b  
#Find the variables that best predict the output
#Filter method using Pearson Correlation  -1<x<1 using heat map, -1 is weak, 1 is strong
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap=plt.cm.Reds) #Seaborn doesnt work with matplotlib<3.3.2 and cant install due to access issues
plt.show()

#Selecting highly correlated features
corr_threshold_lower = 0.2
corr_threshold_upper = 0.9 
corr_target = abs(corrMatrix[targetvar]) #Target variable

relevant_features = corr_target[corr_target>corr_threshold_lower]
relevant_features = relevant_features[corr_target<corr_threshold_upper] #Remove multicolinearity variables
print('Relevant features found:')
print(relevant_features)


#Save features to be removed, so they can iteratively be removed from dataset
not_relevant_features1 = corr_target[corr_target<corr_threshold_lower]
not_relevant_features2 = corr_target[corr_target>corr_threshold_upper]

not_relevant_features = pd.concat([not_relevant_features1, not_relevant_features2], axis=0,sort=False)
print('\n\nNot relevant features:')
print(not_relevant_features)
not_relevant_features = not_relevant_features.index #it is a Pandas series, so need to remove by index



'''Prepare the data for modeling'''
y = df[targetvar] #Target variable
x = df[['mean_Valence','critical_temp']]


'''New Prediction'''
regr = LinearRegression()  
regr.fit(x, y)
y_pred = regr.predict(x)  

#Prepare Results  
results = pd.DataFrame()
#df2['Model Features'] = relevant_features.index
results['Model Features'] = ['mean_Valence','critical_temp']
results['Model Coefficients:'] = regr.coef_
results['Pearson Correlations'] = list(relevant_features[['mean_Valence','critical_temp']])

print('\n\nModel Generation')
print('Bias = ' + str(regr.intercept_) + '\n')
print(results)


ds['Predicted'] = list(y_pred)
ds['Error'] = ( ds['Predicted'] - df[targetvar]) / df[targetvar] 

ds.drop(ds[ds['Error']>3].index, inplace = True)


#Visualize Error
fig = plt.figure(figsize=(5,1)) #Normal Distribution of Error
ax1 = plt.axes(frameon=False) #Hide y_axis
ax1.axes.get_yaxis().set_visible(False)
plt.hist(ds['Error'], bins=50, density=True, alpha=0.6, align='mid',edgecolor='black',color='slategrey')
plt.xticks(fontsize=8)
mu, std = norm.fit(ds['Error'])
xmin, xmax = plt.xlim()
lim = np.linspace(xmin, xmax, 10)
p = norm.pdf(lim, mu, std)
plt.plot(lim, p, 'k', '--',linewidth=2)
plt.title('Error Normal Distribution')
