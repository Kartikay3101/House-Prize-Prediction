"Understanding the Data"
# Import all Important Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing.values())
print(type(housing))
print(housing.DESCR)


"Preparation of Data"
# Creating DataFrame
df=pd.DataFrame(housing.data,columns=housing.feature_names)
df2=df.copy()
df['Prize']=housing.target

# Statistical description of data
df.describe()

# Checking the null values
df.isnull().sum()


"Exploratory Data Analysis"
# Using ProfileReport for better Understanding of Data
from ydata_profiling import ProfileReport
profile=ProfileReport(df,title='EDA Report',explorative=True)
profile.to_file('EDA.html')

# Finding the correlation between the columns
df.corr()

# Using Boxplot to check the outlier
fig, axs = plt.subplots(figsize=(20, 5))
sns.boxplot(data=df, ax=axs)

# Splitting the data into independent and dependent features
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

# Splitting the data into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=42)

# Normalizing the given dataset
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_norm=sc.fit_transform(x_train)
x_test_norm=sc.transform(x_test)


"Model Training"
# Using Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gr=GradientBoostingRegressor()
gr.fit(x_train_norm,y_train)


"Model Pridiction"
reg_pred=gr.predict(x_test_norm)
reg_pred

# Calculate the Residual value
residuals=y_test-reg_pred
residuals

# This plot visually represents the spread and shape of residuals. A symmetric and approximately normal distribution around zero is desirable.
sns.displot(residuals,kind='kde')


"Model Evaluation"
# Its important to Evaluate our model weather its efficient or not 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(f"The Value of MSE is: {mean_squared_error(y_test,reg_pred)}")
print(f"The Value of MAE is: {mean_absolute_error(y_test,reg_pred)}")
print(f"The Value of r2 score is: {r2_score(y_test,reg_pred)}")


"Saving the model"
# We can use pickle to save the file for using the model in further cases
import pickle

# To Save the Model
pickle.dump(gr,open('model.pkl','wb'))

# To Load the Model
model=pickle.load(open('model.pkl','rb'))

# Taking a sample from the data for the model verification
arr=np.array(df2.sample())
arr.flatten()
model.predict(sc.transform(arr.reshape(1,-1)))

