import numpy  as np  
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargamos los datos y dividimos entre train y test.

house_data = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\pr_ml\\data\\airbnb-listings-extract.csv', sep=';', decimal='.')

# Filtramos los datos para la ciudad de Madrid
filtro = house_data['City'] == 'Madrid'
house_data = house_data[filtro]

house_train, house_test = train_test_split(house_data, test_size=0.2, shuffle=True, random_state=0)

print(f'Dimensiones del dataset de training: {house_train.shape}')
print(f'Dimensiones del dataset de test: {house_test.shape}')

house_train.to_csv('C:\\Users\\guill\\OneDrive\\Desktop\\pr_ml\\data\\train.csv', sep=';', decimal='.', index=False)
house_test.to_csv('C:\\Users\\guill\\OneDrive\\Desktop\\pr_ml\\data\\test.csv', sep=';', decimal='.', index=False)

