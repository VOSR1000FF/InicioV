# InicioV
## Curso de bioinformatica
### Datos de tesis de licenciatura 
Dado que hemos llegado a hacer varias maneras de graficar los datos puestos en un dataframe asi como los comandos para pasar de un directorio a otro, copiar, borrar y mover, asi mismo de activar los ambientes de trabajo donde se descargan los paquetes.
Para mi proyecto comienzo con el manejo de mis datos de la tesis de licenciatura.

Mi tesis de licenciatura consistio en verificar que efecto tiene una especie nativa del altiplano como *Baccharis tola* sobre algunos parametros del suelo, durante mi estudio estableci un diseño de estudio en la que puse plantines de esta especie en suelo proveniente de campo y los puse en macetas como tratamiento "*con Baccharis tola* (CBt) y otro tratamiento de macetas sin plantines "*sin Baccharis tola*" (SBt), cada tratamientoc con 7 macetas, saque muestras de suelo con cilindros de metal alrededor de la planta a 15 cm de profundidad, el estudio consistio en 5 meses, el muestreo fue en la ultima semana de cada mes.

Medi algunos parametros como: Actividad Fosfatasa, Glucosidasa como actividad enzimatica, y fosforo y carbono de la biomasa microbiana, asi como actividad respirometrica, estos como parametros microbiologicos, y otros como pH, conductividad electrica como parametros fisicoquimicos.

Primero comienzo con la importacion de librerias

import pandas as cs para el manejo de datos
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter para elaborar lo graficos
import seaborn as sns graficos
import numpy as np 
from scipy.stats import pearsonr
from scipy.stats import shapiro, levene
from scipy.stats import mannwhitneyu
para importar los estadisticos de prueba
from sklearn.decomposition import PCA
para poder realizar el analisis de componentes principales

Luego importo mis datos en llamado "TodoCA.csv" con la direccion del directorio

Hago una revision de la informacion de los datos y la estadistica descriptiva con *.info y *.describe

En esta parte de correccion de espacios, el programa de python me salia error debido a que en los nombres de las columnas tenian espacios entonces lo que hago es eliminar los espacios de cada variable:
tola.columns = tola.columns.str.strip()

Entonces para proseguir comienzo probando un grafico de dispersion para dos variables como "Fosfatasa" y "pH" en funcion de los tratamientos

Sin embargo al tener varias variables me resultaria muy tedioso hacer cada combinacion en un codigo aparte, entonces hize una matriz de dispersion que me adiciono los graficos de dispersion de todas las combinaciones de las variables:
sns.pairplot(tola, vars=vars_of_interest, hue='Tratamiento')
Primero selecciono las variables de interes y los grafico en la matriz en funcion del tratamiento.

Para verificar el efecto de los plantines
Hize graficos de caja para visualizar en que mes del estudio hubo un efecto de los plantines en cada variable, asi que hago graficos de caja para cada variable,aqui emplee la forma de poner subindice dentro de los titulos de los ejes.

Hago matrices de correlación
Luego me interesa hacer correlaciones para saber que variables se relacionan positiva o negativamente entre si, para eso hago una matriz de correlaciones en mapas de calor con el comando:
correlation_matrix = selected_df.corr()
Seleccionando las variables de interes

Asimismo hize uns test de la correlacion donde obtuve una matriz de valores de p-valores donde el mapa de calor marca con mas intensidad valores de p < 0.01 con el comando:
sns.heatmap(p_values_matrix, annot=True, cmap='YlGnBu_r', center=0.01, cbar_kws={'label': 'Valor de p'})

Ahora pruebas de normalidad y homogeneidad

Tenia considerado buscar alguna diferencia significativa segun mis graficos de caja entonces antes de hacer un estadistico hize una prueba de shapiro y levene para verificar si mis datos cumplen con los supuestos, entonces hago estas pruebas para cada variable.

Los resultados me muestran que algunas variables si tienen una distribucion normal (p<0.05), pero todas las variables consideradas no manifiestan varianzas homogeneas (p>0.05).

Entonces decido hacer una prueba no parametrica

Como mis datos no cumplen los supuestos, acudi a la prueba de U-mann-Whitnney como estadistico que obvia los supuestos mencionados. Con el codigo:

En mi base de datos tola considero Tratamiento y Mes y las variables numéricas
Almacenar los resultados

results = []

Obtengo el registro de meses unicos

meses_unicos = tola['Mes'].unique()

Obtengo el el tipo de variables numericas
variables_numericas = tola.select_dtypes(include=['float64', 'int64']).columns

Estadistico de U-Mann-Whitney para cada mes y cada variable en funcion del tratamiento
for mes in meses_unicos:
    for variable in variables_numericas:
        # Filtrar el DataFrame segun la columna 'Mes'
        df_mes = tola[tola['Mes'] == mes]
        
        # Obtener los valores de tratamiento CBt y tratamiento SBt
        tratamiento_a = df_mes[df_mes['Tratamiento'] == 'CBt'][variable]
        tratamiento_b = df_mes[df_mes['Tratamiento'] == 'SBt'][variable]
        
        # Realizar la prueba de U-Mann-Whitney 
        stat, p = mannwhitneyu(tratamiento_a, tratamiento_b)
        
        # Almacenar el resultado en una lista
        results.append({'Mes': mes, 'Variable': variable, 'Statistic': stat, 'p-value': p})
Convertir los resultados a un DataFrame 
results_df = cs.DataFrame(results)

Función para aplicar estilos condicionales 
def highlight_pval(val):
    color = 'red' if val < 0.05 else ''
    return f'background-color: {color}'

Aplicar el estilo condicional a la columna 'p-value'
styled_results_df = results_df.style.applymap(highlight_pval, subset=['p-value'])
Mostrar los resultados 
styled_results_df

La parte de aplicar estilos en tablas es un paquete llamado styled me permite marcar con colores ciertos datos de la tabla de valores de probabilidad cuando cumple cierta condicion "if".

Para esta parte como ultimo analisis hago un PCA para explicar que variables estan relacionadas con los datos tomados.

Tola con 'Tratamiento' y 'Mes', y variables numéricas

Seleccionar solo las variables numéricas para el PCA
variables_numericas = tola.select_dtypes(include=['float64', 'int64'])

Realizar el PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(variables_numericas)

Crear un DataFrame con los resultados del PCA
pca_df = cs.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Tratamiento'] = tola['Tratamiento']

Obtener el porcentaje de varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_ * 100

Crear el gráfico de dispersión
plt.figure(figsize=(12, 8))

Graficar los puntos del PCA, coloreados por tratamiento y agrupados en círculos
sns.scatterplot(x='PC1', y='PC2', hue='Tratamiento', data=pca_df, palette='Set1', s=100)

Añadir las flechas para las variables
for i, var in enumerate(variables_numericas.columns):
    plt.arrow(0, 0, pca.components_[0, i] * np.max(pca_result[:, 0]), pca.components_[1, i] * np.max(pca_result[:, 1]), 
              color='r', alpha=0.5, head_width=0.05)
    plt.text(pca.components_[0, i] * np.max(pca_result[:, 0]) * 1.15, pca.components_[1, i] * np.max(pca_result[:, 1]) * 1.15, 
             var, color='g', ha='center', va='center')

Poner titulo a los ejes
plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA en funcion de los tratamientos')
plt.grid()
Mostrar el gráfico
plt.show()
Mostrar los componentes principales
print(cs.DataFrame(pca.components_, columns=variables_numericas.columns, index=['PC1', 'PC2']))


Me exporta un grafico donde marca cada dato en funcion de los tratamientos y las variables direccionadas en lineas. Graficadas en los componentes principales que explican el mayor porcentaje de varianza.

### Datos preliminares de tesis de maestria
Ahora me encuentro haciendo mi tesis de maestria y vi la oportunidad de aprovechar el uso del programa para hacer algunos analisis de mis datos preliminares.

En lo que consiste mi tesis de maestria, tome muestras de agua en un periodo de 6 meses, en cada mes con dos eventos de muestro en tres sitios diferentes de la ciudad de la paz a traves del Rio Choqueyapu, en cada muestreo tome datos de Temperatura, conductividad, pH y coordenadas en grados decimales.
Luego los lleve al laboratorio y procese las muestras para realizar un conteo de Coliformes totales, *Escherichia coli* en medio de cultivo diferencial Coliformes CPE y *Escherichia coli* CPE en medio de cultivo diferencial y selectivo.

Primero comienzo con la importacion de librerias
mport pandas as cpe
import matplotlib.pyplot as fip
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import folium #para introducir el manejo de coordenadas
import statsmodels.api as sm # para el manejo de modelos lineales
from statsmodels.stats.outliers_influence import variance_inflation_factor # para calcular el VIF

Importo mis datos en formato .csv
carba = cpe.read_csv('/home/siles/comparar/InicioV/Para_Analisis.csv', sep =',') 
carba.head() # mostrar las primeras 5 filas de la base de datos
Uso el codigo info y describe para vizualizar los datos y su estadistica descriptiva

En esta parte, dado que tengo datos de coordenadas, busque una manera de colocarlo en un mapa entonces el paquete folium es el que me salio recomendado para este caso:

import folium
Crear un mapa centrado en la media de las coordenadas
map_center = [carba['Latitud'].mean(), carba['Longitud'].mean()]
mapa = folium.Map(location=map_center, zoom_start=12)

Añadir puntos al mapa
for idx, row in carba.iterrows():
    folium.Marker([row['Latitud'], row['Longitud']]).add_to(mapa)

Guardar el mapa en un archivo HTML
mapa.save("mapa.html")
Lo que me genera es un archivo .html la cual me muestra cada punto de coordenada en los tres sitios muestreados.

Luego pase a elaborar una matriz de dispersion de ciertas variables para visualizar el comportamiento entre ellas, considere tomar pH, CE, Temp, Coltot, Ecoli, RosadosCPE, AzulesCPE

Posteriormente, como en la base de datos de licenciatura hize grafico de cajas, pero ahora en funcion de los sitios de muestreo para cada variable, verificando los resultados los sitios holguin y lipari muestran valores altos en comparacion con Incachaca, que es un sitio donde el agua es limpia y no esta influenciada por la carga urbana.

Despues, acordandome de que lleve modelos lineales en estadistica avanzada, solo que este fue abordado en R studio, busque la posibilidad de hacer una comparacion de modelos lineales usando mis variables numericas, entonces primero para este tipo de analisis hago una matriz de correlacion:
correlation_matrix = selected_df.corr()
¿Porque?
En este tipo de analisis para comenzar, siempre se debe verificar las correlaciones entra las variables predoctoras para evitar multicolinealidad y si no la eliminamos podemos cometer errores en la interpretacion de que variables tienen mas relevancia en el comportamiento de los datos.
Como puedo ver correlaciones de consideracion entre las variables predictorias, hago un calculo del indice inflacion (VIF) con: 

Dentro de mi dataframe selecciono las variables predictoras:
variables_predictoras = ['CE', 'pH', 'Temp', 'Coltot', 'Ecoli']

Creo un dataframe con las variables predictoras
X = carba[variables_predictoras]

se debe hacer una constante para la interceptación en el modelo de regresión
X = sm.add_constant(X)
Calcular el VIF para cada variable
vif_data = cpe.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

EL resultado es el indice VIF, segun interpretandolo un VIF>5 muestra una alta multicolinealidad por tanto es necesario obviar las variables dentro de ese rango, para esta parte el CE y pH mostraron un VIF>5.

Entonces considere quitar CE y pH y hacer una nueva correlacion
carba_reducido = carba.drop(columns=['CE', 'pH'])
carba_reducido
variables_predictoras = ['Temp', 'Coltot', 'Ecoli', 'AzulesCPE', 'RosadosCPE']
correlaciones = carba_reducido[variables_predictoras].corr()
print(correlaciones)

Entonces ahora elaboro cierta cantidad de modelos donde considero a mi variable dependiente como RosadosCPE que consisten en la cantidad de *Escherichia coli* CPE, en funcion de las demas variables.

Hago combinaciones de una, dos y tres variables, asi como el modelo nulo y el global la cual es necesario para la comparacion de modelo.

Luego ya establecidos los valores de AIC hize un calculo del deltaAIC y el peso de Akaike para verificar que modelo se ajusta a los datos y que variables tienen influencia sobre la cantidad de *E.coli* CPE (RosadosCPE)

Creo una lista de los AIC
AICs = [AIC_A, AIC_B, AIC_C, AIC_global, AIC_nulo, AIC_dos_predictores, AIC_dos_predictores1, AIC_dos_predictores2, 
        AIC_dos_predictores3, AIC_dos_predictores4, AIC_tres_predictores1, AIC_tres_predictores2, AIC_tres_predictores3]

Calcular el delta AIC
delta_AIC = [aic - min(AICs) for aic in AICs]

Calcular el peso de Akaike
exp_delta_AIC = [np.exp(-0.5 * delta) for delta in delta_AIC]
sum_exp_delta_AIC = sum(exp_delta_AIC)
peso_Akaike = [val / sum_exp_delta_AIC for val in exp_delta_AIC]

Colocar los resultados en un nuevo dataframe
resultados = cpe.DataFrame({
    'Modelo': ['Coltot', 'ColCPE', 'Temp', 'Global', 'Nulo', 'Coltot + Ecoli total', 
    'ColCPE + Ecoli total','Temp + Ecoli total', 'Coltot + Temp','ColCPE + Coltot',
    'Coltot + Ecoli total + ColiCPE','Coltot + Ecoli total + Temp', 'E.coli total + Temp + ColCPE'],
    'AIC': AICs,
    'Delta AIC': delta_AIC,
    'Peso de Akaike': peso_Akaike
})
resultados