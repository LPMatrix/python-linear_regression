import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('USA_Housing.csv')

X = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']

y = data['Price']

sns.distplot(data['Price'])
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.4, random_state=101)

LR = LinearRegression()

LR.fit(X_train,y_train)

df = pd.read_csv('Ecommerce Customers')
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent']]
y = df['Yearly Amount Spent']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=100)

lrm = LinearRegression()
lrm.fit(X_train,y_train)

predictions = lrm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Test data')
plt.ylabel('Predictions')
plt.show()

sns.heatmap(df.corr(),annot=True)
plt.show()

sns.jointplot(data=df,x='Yearly Amount Spent',y='Time on App')
plt.show()

