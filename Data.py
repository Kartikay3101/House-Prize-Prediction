# Creating DataFrame
df=pd.DataFrame(housing.data,columns=housing.feature_names)
df2=df.copy()
df['Prize']=housing.target
df
# Statistical description of data
df.describe()