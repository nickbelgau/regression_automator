import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import norm


#https://archive.ics.uci.edu/ml/datasets/Stock+portfolio+performance#

file = 'AirQualityUCI.xlsx'
targetvar = 'NOx(GT)'

df = pd.read_excel(file)
ds = df[['Date','Time',targetvar]]
df = df.drop(['Date','Time'], axis=1)

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
corr_threshold_lower = 0.3
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
x = df
for i in not_relevant_features: #Remove the not relevant features
    x = x.drop([str(i)], axis=1) 


'''New Prediction'''
regr = LinearRegression()  
regr.fit(x, y)
y_pred = regr.predict(x)  
  
df2 = pd.DataFrame()
df2['Model Features'] = relevant_features.index
df2['Coefficients:'] = regr.coef_
print('\n\nModel Generation')
print('Bias = ' + str(regr.intercept_) + '\n')
print(df2)


ds['Predicted'] = y_pred.tolist() #Add predicted returns from regression 
ds['Error'] = (ds['Predicted'] - df[targetvar]) / df[targetvar]


#Visualize Prediction
fig = plt.figure(figsize=(5,2))
plt.scatter(ds['Date'],ds[targetvar],color='red',marker=".")
plt.scatter(ds['Date'],ds['Predicted'],color='black',marker=".")
plt.xticks(rotation = 30, fontsize=8)
ax1 = plt.axes()
#ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.title("Target Variable Prediction")

#Visualize Error
fig = plt.figure(figsize=(5,1)) #Normal Distribution of Error
ax1 = plt.axes(frameon=False) #Hide y_axis
ax1.axes.get_yaxis().set_visible(False)
plt.hist(ds['Error'], bins=10, density=True, alpha=0.6, align='mid',edgecolor='black',color='slategrey')
plt.xticks(fontsize=8)
mu, std = norm.fit(ds['Error'])
xmin, xmax = plt.xlim()
lim = np.linspace(xmin, xmax, 10)
p = norm.pdf(lim, mu, std)
plt.plot(lim, p, 'k', '--',linewidth=2)
plt.title('Error Normal Distribution')
