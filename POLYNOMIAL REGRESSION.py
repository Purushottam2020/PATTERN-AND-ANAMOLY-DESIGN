import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datas=pd.read_csv("C:/Users/purus/OneDrive/desktop/Position_Salaries.csv")
# print(datas)

X=datas.iloc[:,1:-1].values
y=datas.iloc[:,-1].values
# print(Y)

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
poly.fit(X_poly,y)

lin2=LinearRegression()
lin2.fit(X_poly,y)
plt.scatter(X,y,color='red')
plt.plot(X,lin.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

plt.scatter(X,y,color='red')
plt.plot(X,lin2.predict(poly.fit_transform(X)),color='blue')
plt.title('polynomial regression')
plt.xlabel('temperature')
plt.ylabel('pressure')
plt.show()

print(lin.predict([[110.0]]))
lin2.predict(poly.fit_transform([[110.0]]))
