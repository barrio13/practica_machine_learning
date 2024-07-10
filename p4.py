import numpy as np  
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargamos los datos
data = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\pr_ml\\data\\train.csv', sep=';', decimal='.')

# Eliminamos las columnas que tienen demasiados valores nulos
data = data.drop(['Host Acceptance Rate', 'Square Feet', 'Has Availability', 'License', 'Jurisdiction Names'], axis=1)

# Eliminamos registros que no tengan sentido como apartamentos sin baño, precio o camas
filtro1 = data['Bathrooms'] > 0
filtro2 = data['Price'] > 0
filtro3 = data['Beds'] > 0
data = data[filtro1 & filtro2 & filtro3]

# Eliminamos las columnas que no nos aportan más información
columns_to_drop = [
    'ID', 'Listing Url', 'Scrape ID', 'Last Scraped', 'Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url',
    'Host ID', 'Host URL', 'Host Name', 'Host Location', 'Host Since', 'Host Thumbnail Url', 'Host Picture Url', 'Market',
    'Zipcode', 'State', 'Smart Location', 'Country Code', 'Country', 'Latitude', 'Longitude', 'Amenities', 'Calendar Updated',
    'Calendar last Scraped', 'First Review', 'Last Review', 'Geolocation', 'Features', 'Host About', 'Experiences Offered',
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Neighbourhood', 'Weekly Price', 'Monthly Price', 'Minimum Nights',
    'Maximum Nights', 'City', 'Host Listings Count', 'Calculated host listings count'
]
data = data.drop(columns=columns_to_drop, axis=1)

# Reemplazamos los datos que faltan con la media para las columnas numéricas y con la moda para las columnas categóricas
# Para las columnas "Security Deposit" y "Cleaning Fee", reemplazamos los valores nulos con 0
data["Host Response Time"] = data["Host Response Time"].fillna(data["Host Response Time"].mode()[0])
data["Host Response Rate"] = data["Host Response Rate"].fillna(data["Host Response Rate"].mean())
data["Host Total Listings Count"] = data["Host Total Listings Count"].fillna(data["Host Total Listings Count"].mean())
data["Bathrooms"] = data["Bathrooms"].fillna(data["Bathrooms"].mode()[0])
data["Bedrooms"] = data["Bedrooms"].fillna(data["Bedrooms"].mode()[0])
data["Beds"] = data["Beds"].fillna(data["Beds"].mode()[0])
data["Price"] = data["Price"].fillna(data["Price"].mean())
data["Security Deposit"] = data["Security Deposit"].fillna(0)
data["Cleaning Fee"] = data["Cleaning Fee"].fillna(0)
data["Review Scores Rating"] = data["Review Scores Rating"].fillna(data["Review Scores Rating"].mean())
data["Reviews per Month"] = data["Reviews per Month"].fillna(data["Reviews per Month"].mean())


# Cargamos los datos
data_test = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\pr_ml\\data\\test.csv', sep=';', decimal='.')


# Reemplazamos los datos que faltan con la media para las columnas numéricas y con la moda para las columnas categóricas
# Para las columnas "Security Deposit" y "Cleaning Fee", reemplazamos los valores nulos con 0
data_test = data_test.copy()
data_test["Host Response Time"] = data_test["Host Response Time"].fillna(data["Host Response Time"].mode()[0])
data_test["Host Response Rate"] = data_test["Host Response Rate"].fillna(data["Host Response Rate"].mean())
data_test["Host Total Listings Count"] = data_test["Host Total Listings Count"].fillna(data["Host Total Listings Count"].mean())
data_test["Bathrooms"] = data_test["Bathrooms"].fillna(data["Bathrooms"].mode()[0])
data_test["Bedrooms"] = data_test["Bedrooms"].fillna(data["Bedrooms"].mode()[0])
data_test["Beds"] = data_test["Beds"].fillna(data["Beds"].mode()[0])
data_test["Price"] = data_test["Price"].fillna(data["Price"].mean())
data_test["Security Deposit"] = data_test["Security Deposit"].fillna(0)
data_test["Cleaning Fee"] = data_test["Cleaning Fee"].fillna(0)
data_test["Review Scores Rating"] = data_test["Review Scores Rating"].fillna(data["Review Scores Rating"].mean())
data_test["Reviews per Month"] = data_test["Reviews per Month"].fillna(data["Reviews per Month"].mean())
data_test["Neighbourhood Group Cleansed"] = data_test["Neighbourhood Group Cleansed"].fillna(data_test["Neighbourhood Cleansed"])
data_test["Room Type"] = data_test["Room Type"].fillna(data["Room Type"].mode()[0])
data_test["Property Type"] = data_test["Property Type"].fillna(data["Property Type"].mode()[0])




# Creamos el mapa de medias para variables categóricas
categorical = ['Host Response Time', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'Property Type', 'Bed Type', 'Room Type', 'Cancellation Policy']
mean_map = {}
for c in categorical:
    mean = data.groupby(c)['Price'].mean()
    data[c] = data[c].map(mean)    
    mean_map[c] = mean



# Aplicamos logaritmo a la columna de precio
data['Log_Price'] = np.log(data['Price'])

# Eliminamos outliers
data = data[data['Bedrooms'] <= 8]
data = data[data['Bathrooms'] <= 6]
data = data[data['Cleaning Fee'] <= 300]
data = data[data['Reviews per Month'] <= 12.5]



# Eliminamos columnas adicionales
columns_to_drop = [
    'Availability 30', 'Availability 60', 'Availability 90', 'Review Scores Accuracy', 'Review Scores Cleanliness',
    'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value', 'Accommodates',
    'Neighbourhood Cleansed'
]
data = data.drop(columns=columns_to_drop, axis=1)

# Generamos nuevas características
data['Bathroom_Bedrooms'] = data['Bathrooms'] * data['Bedrooms']


# print(data.info())
# print(data.head())


#test

# Eliminamos las columnas que tienen demasiados valores nulos
data_test = data_test.drop(['Host Acceptance Rate', 'Square Feet', 'Has Availability', 'License', 'Jurisdiction Names'], axis=1)

# Eliminamos registros que no tengan sentido como apartamentos sin baño, precio o camas
filtro1 = data_test['Bathrooms'] > 0
filtro2 = data_test['Price'] > 0
filtro3 = data_test['Beds'] > 0
data_test = data_test[filtro1 & filtro2 & filtro3]



# Eliminamos las columnas que no nos aportan más información
columns_to_drop = [
    'ID', 'Listing Url', 'Scrape ID', 'Last Scraped', 'Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview',
    'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url',
    'Host ID', 'Host URL', 'Host Name', 'Host Location', 'Host Since', 'Host Thumbnail Url', 'Host Picture Url', 'Market',
    'Zipcode', 'State', 'Smart Location', 'Country Code', 'Country', 'Latitude', 'Longitude', 'Amenities', 'Calendar Updated',
    'Calendar last Scraped', 'First Review', 'Last Review', 'Geolocation', 'Features', 'Host About', 'Experiences Offered',
    'Host Neighbourhood', 'Host Verifications', 'Street', 'Neighbourhood', 'Weekly Price', 'Monthly Price', 'Minimum Nights',
    'Maximum Nights', 'City', 'Host Listings Count', 'Calculated host listings count'
]
data_test = data_test.drop(columns=columns_to_drop, axis=1)


for c in categorical:
    data_test[c] = data_test[c].map(mean_map[c])

print(data_test.iloc[0])


# Aplicamos logaritmo a la columna de precio
data_test['Log_Price'] = np.log(data_test['Price'])

# Eliminamos outliers
data_test = data_test[data_test['Bedrooms'] <= 8]
data_test = data_test[data_test['Bathrooms'] <= 6]
data_test = data_test[data_test['Cleaning Fee'] <= 300]
data_test = data_test[data_test['Reviews per Month'] <= 12.5]

# Eliminamos columnas adicionales
columns_to_drop = [
    'Availability 30', 'Availability 60', 'Availability 90', 'Review Scores Accuracy', 'Review Scores Cleanliness',
    'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value', 'Accommodates',
    'Neighbourhood Cleansed'
]
data_test = data_test.drop(columns=columns_to_drop, axis=1)


# Generamos nuevas características
data_test['Bathroom_Bedrooms'] = data_test['Bathrooms'] * data_test['Bedrooms']

print(data_test.info())
print(data_test.head())


print(data_test.isnull().sum())

# Cambiamos la variable objetivo a la primera posición.
columnas = ['Log_Price'] + [col for col in data if col != 'Log_Price']
data = data[columnas]
data_test = data_test[columnas]


data = data.drop(['Price'] , axis = 1)
data_test = data_test.drop(['Price'] , axis = 1)


from sklearn import preprocessing

# Dataset de train
data_train = data.values
y_train = data_train[:,0:1]     
X_train = data_train[:,1:]      

# Dataset de test
data_test_val = data_test.values
y_test = data_test_val[:,0:1]     
X_test = data_test_val[:,1:]      


# Escalamos
scaler = preprocessing.StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)
XtestScaled = scaler.transform(X_test) 


print('Datos entrenamiento: ', XtrainScaled.shape)
print('Datos test: ', XtestScaled.shape)


# Lasso.

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

alpha_vector = np.logspace(-1,10,20)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 3, verbose=2)
grid.fit(XtrainScaled, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('3-Fold MSE')
plt.show()

from sklearn.metrics import mean_squared_error

alpha_optimo = grid.best_params_['alpha']
lasso = Lasso(alpha = alpha_optimo).fit(XtrainScaled,y_train)

ytrainLasso = lasso.predict(XtrainScaled)
ytestLasso  = lasso.predict(XtestScaled)
mseTrainModelLasso = mean_squared_error(y_train,ytrainLasso)
mseTestModelLasso = mean_squared_error(y_test,ytestLasso)

print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso)
print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso)

print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso))

feature_names = data.columns[1:]

w = lasso.coef_
for f,wi in zip(feature_names,w):
    print(f,wi)

# best mean cross-validation score: -0.183
# best parameters: {'alpha': 0.1}
# MSE Modelo Lasso (train): 0.183
# MSE Modelo Lasso (test) : 0.177
# RMSE Modelo Lasso (train): 0.427
# RMSE Modelo Lasso (test) : 0.42


# Este modelo nos dice que las variables más relevantes son Room Type, Bedrooms, Neighbourhood Group Cleansed,Cleaning Fee,Bathroom_Bedrooms.

# Generamos una nueva característica
data['Room_Type_Bedrooms'] = data['Room Type'] * data['Bedrooms']
data_test['Room_Type_Bedrooms'] = data_test['Room Type'] * data_test['Bedrooms']


# Random Forest.

# from sklearn.ensemble import RandomForestRegressor


# y_train = np.ravel(y_train)
# maxDepth = range(1,15)
# tuned_parameters = {'max_depth': maxDepth}

# grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'), param_grid=tuned_parameters,cv=3, verbose=2) 
# grid.fit(X_train, y_train)

# print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
# print("best parameters: {}".format(grid.best_params_))

# scores = np.array(grid.cv_results_['mean_test_score'])
# plt.plot(maxDepth,scores,'-o')
# plt.xlabel('max_depth')
# plt.ylabel('10-fold ACC')

# plt.show()


# maxDepthOptimo = grid.best_params_['max_depth']
# randomForest = RandomForestRegressor(max_depth=maxDepthOptimo,n_estimators=200,max_features='sqrt').fit(X_train,y_train)

# print("Train: ",randomForest.score(X_train,y_train))
# print("Test: ",randomForest.score(X_test,y_test))



# importances = randomForest.feature_importances_
# importances = importances / np.max(importances)

# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10,10))
# plt.barh(range(X_train.shape[1]),importances[indices])
# plt.yticks(range(X_train.shape[1]),feature_names[indices])
# plt.show()

# best mean cross-validation score: 0.746
# best parameters: {'max_depth': 14}
# Train:  0.8727981241224567
# Test:  0.7849075560856553

# Selección de características, eliminamos variables con menos relevancia.
data = data.drop(['Host Response Time','Bed Type'] , axis = 1)
data_test = data_test.drop(['Host Response Time','Bed Type'], axis = 1)



# Dataset de train
data_train = data.values
y_train = data_train[:,0:1]     
X_train = data_train[:,1:]      

# Dataset de test
data_test_val = data_test.values
y_test = data_test_val[:,0:1]     
X_test = data_test_val[:,1:]      

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


y_train = np.ravel(y_train)
maxDepth = range(1,15)
tuned_parameters = {'max_depth': maxDepth}

grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'), param_grid=tuned_parameters,cv=3, verbose=2) 
grid.fit(X_train, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth')
plt.ylabel('10-fold ACC')

plt.show()


maxDepthOptimo = grid.best_params_['max_depth']
randomForest = RandomForestRegressor(max_depth=maxDepthOptimo,n_estimators=200,max_features='sqrt').fit(X_train,y_train)

print("Train: ",randomForest.score(X_train,y_train))
print("Test: ",randomForest.score(X_test,y_test))



importances = randomForest.feature_importances_
importances = importances / np.max(importances)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,10))
plt.barh(range(X_train.shape[1]),importances[indices])
plt.yticks(range(X_train.shape[1]),feature_names[indices])
plt.show()


# best mean cross-validation score: 0.745
# best parameters: {'max_depth': 14}
# Train:  0.8703200610730477
# Test:  0.7825764588362943




# Los valores de MSE con el modelo Lasso son bajos por lo que tiene que estar haciendo buenas predicciones y la diferencia entre el MSE de train y el MSE de test son pequeños lo que quiere decir que está generalizando bien.
# Con Random Forest podemos ir haciendo selección de características para ver si los resultados van mejorando, podemos volver a probar Lasso para ver si mejora con un modelo simplificado.
# Tenemos una diferencia de un 10% entre el train y test y el objetivo sería reducirla.
# Podemos buscar también generar nuevas características a partir de los datos que nos proporciona Random Forest, una combinación lineal de características. O porbar con nuevos modelos.
