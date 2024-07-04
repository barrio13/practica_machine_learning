import numpy  as np  
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuramos pandas para mostrar más columnas y filas.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  


data = pd.read_csv('C:\\Users\\guill\\OneDrive\\Desktop\\pr_ml\\data\\train.csv', sep=';', decimal='.')

#Hacemos un ánalisis exploratorio.

# print(data.describe())
# print(data.info())
print(data.isnull().any())


print(f"Cantidad de registros antes del filtrado: {len(data)}")
print(f"Cantidad de registros después del filtrado: {len(data)}")

# Calculamos la cantidad de valores nulos en cada columna.
null_counts = data.isnull().sum()
print("Cantidad de valores nulos por columna:")
print(null_counts)

# Eliminamos las columnas que tienen demasiados nulos.
data = data.drop(['Host Acceptance Rate','Square Feet','Has Availability','License','Jurisdiction Names'], axis=1)

#Eliminamos registros que no tengan sentido como apartamentos sin baño, precio o camas.

filtro1 = data['Bathrooms'] > 0
filtro2 = data['Price'] > 0
filtro3 = data['Beds'] > 0
data_madrid = data[filtro1 & filtro2 & filtro3]

# Eliminamos las columnas que no nos aportan más información.

data_madrid = data_madrid.drop(['ID','Listing Url','Scrape ID','Last Scraped','Name','Summary','Space','Description','Neighborhood Overview','Notes','Transit','Access','Interaction','House Rules'
                                ,'Thumbnail Url','Medium Url','Picture Url','XL Picture Url','Host ID','Host URL','Host Name','Host Location','Host Since','Host Thumbnail Url',
                                'Host Picture Url','Market','Zipcode','State','Smart Location','Country Code','Country','Latitude','Longitude','Amenities','Calendar Updated','Calendar last Scraped',
                                 'First Review','Last Review','Geolocation','Features','Host About','Experiences Offered','Host Neighbourhood',
                                'Host Verifications','Street','Neighbourhood','Weekly Price','Monthly Price','Minimum Nights','Maximum Nights','City','Host Listings Count','Calculated host listings count'
                                ,], axis=1)
print(data_madrid.info())
print(data_madrid.iloc[1])
print(data_madrid.isnull().any())

# Reemplazamos los datos que faltan con la media para las columnas numéricas y con la moda para las columnas con clases. Para las columnas "Security Deposit" y "Cleaning Fee" se puede
# entender la falta de valor como que no se pide, al ser opcionales reemplazamos con 0.


data_madrid["Host Response Time"].fillna(data_madrid["Host Response Time"].mode()[0], inplace=True)
data_madrid["Host Response Rate"].fillna(data_madrid["Host Response Rate"].mean(), inplace=True)
data_madrid["Host Total Listings Count"].fillna(data_madrid["Host Total Listings Count"].mean(), inplace=True)
data_madrid["Bathrooms"].fillna(data_madrid["Bathrooms"].mode()[0], inplace=True)
data_madrid["Bedrooms"].fillna(data_madrid["Bedrooms"].mode()[0], inplace=True)
data_madrid["Beds"].fillna(data_madrid["Beds"].mode()[0], inplace=True)
data_madrid["Price"].fillna(data_madrid["Price"].mean(), inplace=True)
data_madrid["Security Deposit"].fillna(0, inplace=True)
data_madrid["Cleaning Fee"].fillna(0, inplace=True)
data_madrid["Review Scores Rating"].fillna(data_madrid["Review Scores Rating"].mean(), inplace=True)
data_madrid["Review Scores Accuracy"].fillna(data_madrid["Review Scores Accuracy"].mean(), inplace=True)
data_madrid["Review Scores Cleanliness"].fillna(data_madrid["Review Scores Cleanliness"].mean(), inplace=True)
data_madrid["Review Scores Checkin"].fillna(data_madrid["Review Scores Checkin"].mean(), inplace=True)
data_madrid["Review Scores Communication"].fillna(data_madrid["Review Scores Communication"].mean(), inplace=True)
data_madrid["Review Scores Location"].fillna(data_madrid["Review Scores Location"].mean(), inplace=True)
data_madrid["Review Scores Value"].fillna(data_madrid["Review Scores Value"].mean(), inplace=True)
data_madrid["Reviews per Month"].fillna(data_madrid["Reviews per Month"].mean(), inplace=True)

# Ahora para todas las variables catégoricas creamos un mapa con sus medias respecto al precio.

# Vemos cúales son las variables catégoricas.
print(data_madrid.dtypes)
print(data_madrid.apply(lambda x: len(x.unique())))

# Creamos el mapa.

categorical = ['Host Response Time','Neighbourhood Cleansed', 'Neighbourhood Group Cleansed','Property Type','Bed Type','Room Type','Cancellation Policy']

mean_map = {}
for c in categorical:
    mean = data_madrid.groupby(c)['Price'].mean()
    data_madrid[c] = data_madrid[c].map(mean)    
    mean_map[c] = mean


print(data_madrid.iloc[1])

# Vamos a ver si existen datos anómalos.

plt.figure(figsize=(15, 5))

data_madrid['Price'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.yscale("log")
plt.xlabel('Price')

plt.show()

# Vemos que podemos aplicar el logaritmo a la columna del Precio.
data_madrid['Log_Price'] = np.log(data_madrid['Price'])

# Graficamos las relaciones
columnas_verificacion = ['Bathrooms', 'Bedrooms', 'Beds', 'Security Deposit', 'Cleaning Fee', 'Availability 365', 'Review Scores Rating', 'Reviews per Month']

for col in columnas_verificacion:
    data_madrid.plot(kind='scatter', x=col, y='Log_Price')
    plt.xlabel(col)
    plt.ylabel('Log_Price')
    plt.show()


#Eliminación outliers

data_madrid = data_madrid[data_madrid['Bedrooms'] <= 8]
data_madrid = data_madrid[data_madrid['Bathrooms'] <= 6]
data_madrid = data_madrid[data_madrid['Cleaning Fee'] <= 300]
data_madrid = data_madrid[data_madrid['Reviews per Month'] <= 12.5]


# Ahora tenemos todas las columnas con valores númericos y podemos ver la correlación que tienen.

corr = np.abs(data_madrid.drop(['Log_Price'], axis=1).corr())
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})
plt.show()


# De aquí podemos sacar que no necesitamos las siguientes columnas.

data_madrid = data_madrid.drop(['Availability 30','Availability 60','Availability 90','Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin',
                                'Review Scores Communication','Review Scores Location','Review Scores Value','Accommodates','Neighbourhood Cleansed'], axis=1)


# Generamos nuevas características.
data_madrid['Bathroom_Bedrooms'] = data_madrid['Bathrooms'] * data_madrid['Bedrooms']
data_madrid['Total_Price'] = data_madrid['Price'] - data_madrid['Cleaning Fee']

# print(data_madrid.shape)

# pd.plotting.scatter_matrix(data_madrid, alpha=0.2, figsize=(20, 20), diagonal = 'kde')
# plt.show()

