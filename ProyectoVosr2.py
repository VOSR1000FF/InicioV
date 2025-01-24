#!/usr/bin/env python
# coding: utf-8

# # *Manejo de datos preliminares de tesis de Maestria*
# ## En este archivo pongo los analisis que pude hacer para mis datos que reuni hasta el momento en mi tesis de maestria.
# - En el README esta mas a detalle que hize en mi estudio pero para recordarlo hize un estudio en 12 meses de recolecta de muestras de agua residual, aisle bacterias CPE para los cuales hize caracterizacion con pruebas fenotipicas.
# - Ademas tome otros datos mas de las muestras de agua para tener mas cobertura de estudio.

# In[1]:


# Para empezar hago una entrada de las librerias
import pandas as cpe
import matplotlib.pyplot as fip
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import folium
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


# Importo mis datos en formato .csv
carba = cpe.read_csv('/home/siles/comparar/InicioV/Para_Analisis.csv', sep =',') 
carba.head() # mostrar las primeras 5 filas de la base de datos


# In[3]:


carba.info() #informacion de los datos
carba.describe() #Est descriptivos


# Dado que mis datos contienen coordenadas en decimales quize colocarlos en un mapa para mostrar los tres puntos de muestreo
# donde tome muestras de agua a lo largo del Rio Choqueyapu

# In[10]:


import folium

# Crear un mapa centrado en la media de las coordenadas
map_center = [carba['Latitud'].mean(), carba['Longitud'].mean()]
mapa = folium.Map(location=map_center, zoom_start=12)

# Añadir puntos al mapa
for idx, row in carba.iterrows():
    folium.Marker([row['Latitud'], row['Longitud']]).add_to(mapa)

# Guardar el mapa en un archivo HTML
mapa.save("mapa.html")


# ## Ahora como en la otra base de datos, aqui hago directamente un grafico de dispersion de todas las variables

# In[4]:


# Convertir las columnas a notación científica y reemplazar los valores originales
columnas_a_convertir = ['Coltot', 'Ecoli', 'AzulesCPE', 'RosadosCPE']
for columna in columnas_a_convertir:
    carba[columna] = carba[columna].apply(lambda x: '{:.2e}'.format(x) if cpe.notnull(x) else x)
carba


# In[6]:


# Convertir columnas a formato numérico
columnas_a_convertir = ['Coltot', 'Ecoli', 'AzulesCPE', 'RosadosCPE']
for columna in columnas_a_convertir:
    carba[columna] = cpe.to_numeric(carba[columna], errors='coerce')

vars_of_interest = ['CE', 'pH', 'Temp', 'Coltot', 'Ecoli', 'AzulesCPE', 'RosadosCPE']
# Las columnas de interés son numéricas
for columna in columnas_a_convertir:
    carba[columna] = carba[columna].apply(lambda x: float('{:.2e}'.format(x)) if cpe.notnull(x) else x)

# Verifico que no haya valores nulos después de la conversión
print(carba[columnas_a_convertir].isnull().sum())

# Elimino valores nulos en las columnas de interés
carba = carba.dropna(subset=vars_of_interest)

# Exporto una matriz de gráficos de dispersión
sns.pairplot(carba, vars=vars_of_interest, hue='Muestra')

# Mostrar el gráfico
fip.show()


# ## En esta parte hago graficos de caja para cada variable en funcion de los tres sitios de muestreo

# In[7]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['cyan', 'yellow', 'orange'] 
sns.boxplot(data=carba, x='Muestra', y='pH', hue='Muestra', palette=colores_personalizados, legend=False) 

def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('pH en cada sitio ')
fip.xlabel('Sitios') 
fip.ylabel('[pH]') 
fip.show()


# In[8]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['blue', 'green', 'yellow'] 
sns.boxplot(data=carba, x='Muestra', y='CE', hue='Muestra', palette=colores_personalizados, legend=False) 

def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('Conductividad electrica ')
fip.xlabel('Puntos de muestreo') 
fip.ylabel('CE [µS]') 
fip.show()


# In[9]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['indigo', 'lime', 'darkblue'] 
sns.boxplot(data=carba, x='Muestra', y='Temp', hue='Muestra', palette=colores_personalizados, legend=False) 

def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('Temperatura de agua')
fip.xlabel('Puntos de muestreo') 
fip.ylabel('Temperatura[°C]') 
fip.show()


# In[10]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['powderblue', 'darkturquoise', 'teal'] 
ax = sns.boxplot(data=carba, x='Muestra', y='Coltot', hue='Muestra', palette=colores_personalizados, legend=False) 
ax.set_ylim(0, 0.5e8)
def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('Coliformes totales')
fip.xlabel('Puntos de muestreo') 
fip.ylabel('[CFU/100ml]') 
fip.show()


# In[11]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['lightgreen', 'lawngreen', 'yellowgreen'] 
sns.boxplot(data=carba, x='Muestra', y='Ecoli', hue='Muestra', palette=colores_personalizados, legend=False) 

def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('Escherichia coli')
fip.xlabel('Puntos de muestreo') 
fip.ylabel('[CFU/100ml]') 
fip.show()


# In[12]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['blue', 'green', 'yellow'] 
sns.boxplot(data=carba, x='Muestra', y='AzulesCPE', hue='Muestra', palette=colores_personalizados, legend=False) 

def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('Coliformes productores de carbapenemasas ')
fip.xlabel('Puntos de muestreo') 
fip.ylabel('[CFU/100ml]') 
fip.show()


# In[13]:


fip.figure(figsize=(12, 6))
colores_personalizados = ['peru', 'red', 'tomato'] 
sns.boxplot(data=carba, x='Muestra', y='RosadosCPE', hue='Muestra', palette=colores_personalizados, legend=False) 

def my_formatter(x, _): 
    return carba['Muestra'].unique()[int(x) % len(carba['Muestra'].unique())] 
fip.gca().xaxis.set_major_formatter(FuncFormatter(my_formatter))

# Personalizar el gráfico 
fip.title('Escherichia coli CPE', fontsize=14, fontstyle='italic')
fip.xlabel('Puntos de muestreo') 
fip.ylabel('[CFU/100ml]')  
fip.show()


# Ahora hago una matriz de correlacion entre las diferentes variables

# In[14]:


# En mi base de datos 'carba'

# Selecciono las variables específicas
selected_vars = ['CE', 'pH', 'Temp', 'Coltot', 'Ecoli', 'AzulesCPE', 'RosadosCPE']  
# Creo un DataFrame solo con las variables seleccionadas
selected_df = carba[selected_vars]

# Calcular la matriz de correlación para las variables seleccionadas
correlation_matrix = selected_df.corr()

# Tamaño del grafico
fip.figure(figsize=(10, 8))

# Matriz de correlacion en mapas de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)

# Mostrar el gráfico
fip.show()


# Segun esta matriz de correlacion algunas variables estan fuertemente correlacionadas, para mis intenciones de que querer hacer modelos lineales, es necesario hallar el factor de influenza de varianza(VIF) y explicar la relacion entre las variables predictorias que en este caso estoy considerando por el momento los datos de conteo de colonias en SuperCARBA de Escherichia coli CPE.

# In[15]:


# Dentro de mi dataframe selecciono las variables predictoras:
variables_predictoras = ['CE', 'pH', 'Temp', 'Coltot', 'Ecoli']
# Creo un dataframe con las variables predictoras
X = carba[variables_predictoras]
# se debe hacer una constante para la interceptación en el modelo de regresión
X = sm.add_constant(X)
# Calcular el VIF para cada variable
vif_data = cpe.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# In[16]:


# Entonces considero quitar CE y pH y hago las correlaciones
carba_reducido = carba.drop(columns=['CE', 'pH'])
carba_reducido
variables_predictoras = ['Temp', 'Coltot', 'Ecoli', 'AzulesCPE', 'RosadosCPE']
correlaciones = carba_reducido[variables_predictoras].corr()
print(correlaciones)


# Ahora como quite algunas variables segun el valor VIF, voy a hacer una comparacion de modelos mediante modelos lineales con AIC, en este caso usare los datos de RosadoCPE para variable independiente y los demas como valores predictoras. Argumentando de como la temperatura, la cantidad de coliformes totales, coliformesCPe y la cantidad de ecoli totales afecta en la cantidad de EcoliCPE.

# In[17]:


# Antes de comenzar el programa me salia error porque mis datos no los veia como numericos entonces uso este comando
# para transformarlo
carba_reducido['Coltot'] = cpe.to_numeric(carba_reducido['Coltot'], errors='coerce')
carba_reducido['AzulesCPE'] = cpe.to_numeric(carba_reducido['AzulesCPE'], errors='coerce')
carba_reducido['Ecoli'] = cpe.to_numeric(carba_reducido['Ecoli'], errors='coerce')
carba_reducido['Temp'] = cpe.to_numeric(carba_reducido['Temp'], errors='coerce')
# ahora creo modelos para una sola variable predictora
modelo_A = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Coltot']])).fit()
modelo_B = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['AzulesCPE']])).fit()
modelo_C = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Ecoli']])).fit()
# Modelos de dos variables predictoras
modelo_dos_predictores = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Coltot', 'Ecoli']])).fit()
modelo_dos_predictores1 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['AzulesCPE', 'Ecoli']])).fit()
modelo_dos_predictores2 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Temp', 'Ecoli']])).fit()
modelo_dos_predictores3 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Coltot', 'Temp']])).fit()
modelo_dos_predictores4 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['AzulesCPE', 'Coltot']])).fit()
# Modelos con tres variables predictoras
modelo_tres_predictores1 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Coltot', 'Ecoli', 'AzulesCPE']])).fit()
modelo_tres_predictores2 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Coltot', 'Ecoli', 'Temp']])).fit()
modelo_tres_predictores3 = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Ecoli', 'Temp', 'AzulesCPE']])).fit()
# Crear el modelo global
modelo_global = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(carba_reducido[['Coltot', 'Ecoli','AzulesCPE','Temp']])).fit()
# Crear el modelo nulo
modelo_nulo = sm.OLS(carba_reducido['RosadosCPE'], sm.add_constant(cpe.Series([1]*len(carba_reducido)))).fit()
# Calcular AIC para cada modelo
AIC_A = modelo_A.aic
AIC_B = modelo_B.aic
AIC_C = modelo_C.aic
AIC_global = modelo_global.aic
AIC_nulo = modelo_nulo.aic 

AIC_dos_predictores = modelo_dos_predictores.aic
AIC_dos_predictores1 = modelo_dos_predictores1.aic
AIC_dos_predictores2 = modelo_dos_predictores2.aic
AIC_dos_predictores3 = modelo_dos_predictores3.aic
AIC_dos_predictores4 = modelo_dos_predictores4.aic

AIC_tres_predictores1 = modelo_tres_predictores1.aic
AIC_tres_predictores2 = modelo_tres_predictores2.aic
AIC_tres_predictores3 = modelo_tres_predictores3.aic

print("AIC Para Coliformes totales:", AIC_A)
print("AIC Coliformes CPE:", AIC_B)
print("AIC Escherichia coli totales:", AIC_C)
print("AIC para el modelo global:", AIC_global)
print("AIC para el modelo nulo:", AIC_nulo)

print("AIC para Coliformes totales + Escherichia coli totales:", AIC_dos_predictores)
print("AIC para Coliformes CPE + Escherichia coli totales:", AIC_dos_predictores1)
print("AIC para Temperatura y Escherichia coli totales:", AIC_dos_predictores2)
print("AIC para Coliformes totales + Temperatura:", AIC_dos_predictores3)
print("AIC para Coliformes CPE + Coliformes totales:", AIC_dos_predictores4)

print("AIC para Coliformes totales + E.coli totales + Coliformes CPE:", AIC_tres_predictores1)
print("AIC para Coliformes totales + E.coli totales + Temperatura:", AIC_tres_predictores2)
print("AIC para E.coli totales + Temperatura + Coliformes CPE:", AIC_tres_predictores3)


# In[18]:


# Creo una lista de los AIC
AICs = [AIC_A, AIC_B, AIC_C, AIC_global, AIC_nulo, AIC_dos_predictores, AIC_dos_predictores1, AIC_dos_predictores2, 
        AIC_dos_predictores3, AIC_dos_predictores4, AIC_tres_predictores1, AIC_tres_predictores2, AIC_tres_predictores3]

# Calcular el delta AIC
delta_AIC = [aic - min(AICs) for aic in AICs]

# Calcular el peso de Akaike
exp_delta_AIC = [np.exp(-0.5 * delta) for delta in delta_AIC]
sum_exp_delta_AIC = sum(exp_delta_AIC)
peso_Akaike = [val / sum_exp_delta_AIC for val in exp_delta_AIC]

# Colocar los resultados en un nuevo dataframe
resultados = cpe.DataFrame({
    'Modelo': ['Coltot', 'ColCPE', 'Temp', 'Global', 'Nulo', 'Coltot + Ecoli total', 
    'ColCPE + Ecoli total','Temp + Ecoli total', 'Coltot + Temp','ColCPE + Coltot',
    'Coltot + Ecoli total + ColiCPE','Coltot + Ecoli total + Temp', 'E.coli total + Temp + ColCPE'],
    'AIC': AICs,
    'Delta AIC': delta_AIC,
    'Peso de Akaike': peso_Akaike
})
resultados


# Segun la interpretacion del deltaAIC cuando es < 2 el modelo se ajusta mejor a los datos, para las variables medidas el modelo global con todas las variables afectan en la cantidad de Ecoli CPE (deltaAIC = 1.87), en tanto el modelo de ColiformesCPE mas E.coli totales y de temperatura mas Ecoli totales tienen influencia sobre la cantidad de Ecoli CPE. Asi mismo para el peso de akaike (w) los mismo modelos mencinados cubren la mayoria de los datos. como en todo analisis de este tipo el modelo nulo debe ser el que tenga un deltaAIC mas alto (deltaAIC = 64.11)
