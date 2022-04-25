from smtplib import LMTP, LMTP_PORT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Q1
# Display the data types of each column using the attribute dtypes, 
# then take a screenshot and submit it, include your code in the image.

file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
df.head()
print(df.dtypes)

# Q2
# Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), 
# then use the method describe() to obtain a statistical summary of the data. 
# Take a screenshot and submit it, make sure the inplace parameter is set to True. 
# Your output should look like this

df=pd.read_csv(file_name)
df.drop(["id", "Unnamed: 0"], axis=1, inplace = True)
print(df.describe())
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

# Q3
# use the method value_counts to count the number of houses with unique floor values, 
# use the method .to_frame() to convert it to a dataframe. 

df['floors'].value_counts()
print(df['floors'].value_counts().to_frame())

# Q4
# use the function boxplot in the seaborn library to produce a plot that can be used to
#  determine whether houses with a waterfront view or without a waterfront view have more price outliers. 
# Your output should look like this with the code that produced it (the colors may be different ) :

sns.boxplot(x="waterfront", y="price", data=df)
plt.show()

# Q5
# Use the function regplot in the seaborn library to determine if the feature 
# sqft_above is negatively or positively correlated with price. 
# Take a screenshot of the plot and the code used to generate it.

sns.regplot(x="sqft_above", y="price", data=df, ci = None)
plt.show()

# Q6
# Fit a linear regression model to predict the price using the feature 'sqft_living' then calculate the R^2. 
# Take a screenshot of your code and the value of the R^2.

X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
print(lm.score(X, Y))

# Q7
# Fit a linear regression model to predict the 'price' using the list of features:

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
A = df[features]
B = df['price']
lm = LinearRegression()
lm.fit(A,B)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)
print(lm.score(A,B))

# Q8 
# Create a pipeline object that scales the data performs a polynomial transform and fits a linear regression model. Fit the object using the features in the question above, then fit the model and calculate the R^2. Take a screenshot of your code and the R^2.

pipe=Pipeline([('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())])
pipe.fit(A,B)
print(pipe.score(A,B))

# Q9
# Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1 and calculate the R^2 using the test data. 
# Take a screenshot for your code and the R^2

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])
x_test_pr=pr.fit_transform(x_test[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr, y_train)
print(RidgeModel.score(x_train_pr, y_train))

# Q10
# Perform a second order polynomial transform on both the training data and testing data. 
# Create and fit a Ridge regression object using the training data, setting the 
# regularisation parameter to 0.1. Calculate the R^2 utilising the test data provided. 
# Take a screenshot of your code and the R^2.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
print(RigeModel.score(x_test_pr, y_test))

