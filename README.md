# ``` | ✏️ Proyecto Integrador Machine Learning ✏️ |🟢 FastApi 🟢|🔵 Render 🔵| 🔴 Sistema de recomendación de películas 🔴 | 🚀 Henry 🚀 ```
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%2370399F.svg?style=for-the-badge&logo=seaborn&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=ffffff)
![venv](https://img.shields.io/badge/Virtualenv-venv-%2300FFFF?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Render](https://img.shields.io/badge/Render-46E3B7.svg?style=for-the-badge&logo=Render&logoColor=white)


![Barner](/assets/barner.png)

# Puedes consultar el proyecto deployado aquí: https://movierecomendationsystem.onrender.com/

# ✔️``Introducción``

Actualmente las máquinas se están haciendo presentes en nuestro día a día, su capacidad de aprendizaje y de predicción que realizan se incrementa a lo largo del tiempo, el surgimiento de los asistentes virtuales hoy día hacen más eficientes la ejecición y solución de tareas más complejas, esto sucede gracias a la mejora de sus algoritmos mediante técnicas de Machine Learning (ML). Un ejemplo de aplicación se da en las plataformas de 'streaming' que utilizan algoritmos para personalizar contenidos mediante el reconocimiento de patrones y similitudes. De esta última idea surge el desarrollo del presente proyecto, un sistema de recomendación de películas que para un buen funcionamiento se lleva a cabo mediante el seguimiento de un proceso sistemático, mismo proceso que rige la carrera de un data Scientist. Para desarrollarlo se requieren contar con los datos necesarios, posteriormente el tratamiento de estos mediante un proceso de ETL (Extracción Transformación y Carga) seguido de un Análisis Exploratorio de Datos para que ingresen al Modelo de Machine Learnin, en donde se llevará a cabo la recomendación de acuerdo a ciertas características dadas. Además, el proyecto consiste en el desarrollo de una API con FastAPI para consultar y recomendar películas, así como otros datos importantes a partir de un conjunto de datos detallado. Se utilizan técnicas de procesamiento de texto y similitud de coseno para generar recomendaciones de películas, y se han implementado varias funcionalidades para obtener información sobre películas, directores y actores.



# ✔️```Los Datos```
Antes de comenzar a describir y explicar el proceso y el funcionamiento del proyecto, se hace notar que los datos originales están alojados en la nube de Google drive, los datos están divididos en dos archivos en formato csv. Debido a que ocupan una gran cantidad de espacio en memoria, se opta por proporcionar el enlace en este repositorio, por lo que cualquier replicación de este proyecto, necesitará descargar y colocar los datos en la carpeta data.
Se adjuntan los enlaces de dichos datos:
### **Fuente de datos**
+ [Dataset](https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5?usp=drive_link): Carpeta con los 2 archivos con datos que requieren ser procesados (movies_dataset.csv y credits.csv), tengan en cuenta que hay datos que están anidados (un diccionario o una lista como valores en la fila).
+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit#gid=0): Diccionario con algunas descripciones de las columnas disponibles en el dataset.



# ✔️```Objetivos del proyecto```

- Implementar un sistema de recomendación de películas basado en Machine Learning.

- Facilitar la disposición de datos en la web mediante creación y consultas de API´s.


# ✔️```Proceso del proyecto```
El siguiente gráfico muestra cual fue el desarrollo del proyecto: 
![Esquema gráfico](/assets/procesamiento.png)



1. ## ⚙️ ETL 
Durante esta etapa del proyecto, de llevó a cabo la extracción, transformación y carga de los datos, todas las transformaciones que se creyeron pertinentes se encuentran en el archivo ``ETL_1.ipybn`` y ``ETL_2.ipybn`` ejecutados en ese orden y ambos archivos alojados en la carpeta ETL. Aquelals transformaciones están documentadas paso a paso dentro del archivo.
> [!IMPORTANT]
> En caso de replicar, es importante ejecutar los archivos de acuerdo al orden de numeración. 👀
### ✔️```Resultados```
![ETL](/assets/ETL.png)


2. ## 📉 EDA
Tras haber completado las tareas de ETL, se realiza el EDA (Análisis Exploratorio de Datos) donde el principal objetivo perseguido es entender como están estructurados los datos, esto para determinar cuales son las características que serán funcionales para nuestro modelo de entrenamiento, dentro de él se encuentran todos aquellos gráficos que ayudan a interpretar las características y su relación entre los mismos, de esta manera, una vez realizado el EDA, se recogen los datos y se llevan al modelo de entrenamiento, hay que recalcar que dichos datos se guardan en formato parquet. El archivo ``EDA.ipynb`` ubicado dentro de la carpeta ``EDA`` contiene los pasos detallados de este punto del proceso.
> [!IMPORTANT]
> Es importante ejecutar este archivo después de haber ejecutado el ETL y haberse asegurado de que se creó el archivo primordial para su ejecución. 👀
### ✔️```Resultados```
![EDA](/assets/EDA.png)


3. ## 🧠 Machine Learning
Una vez llevados a cabo los pasos anteriores se diseña el algoritmo de Machin Learning, mismo que también se encuentra en el archivo ``main.py``. Este algoritmo es el más importante de la aplicación puesto que hace las recomendaciones de películas a los usuarios con el simple hecho de proporcionar información de una película que se encuentre entre los datos.
El Machine Learning consta de 2 pasos indispensables para ser llevado a cabo; 
    - El primero, realizar un correcto EDA para conocer los datos y elegir las variables a implementar en el algoritmo; 
    - El segundo, un correcto entrenamiento del modelo para elegir el idóneo que realiza la tarea con mejor eficiencia, este entrenamiento se puede consultar en el archivo ``recomendationsystem.ipynb`` ubicado en la carpeta raíz del proyecto. 


4. ## ✏️ Desarrollo de API
Una vez que los datos estén procesados y se tenga un entendimiento de los mismos, se realizan las funciones API, la cuales utiliza el usuario para obtener información primordial desplegada en la pantalla de su dispositivo. Las funciones desarrolladas se encuentran dentro del archivo ``main.py``, este archivo será el encargado de desplegar la app en Render, dicho archivo se encuentra en la carpeta raíz de la aplicación y contiene 7 funciones importantes:
        1. cantidad_filmaciones_mes que devuelve la cantidad de películas que fueron entrenadas en el mes consultado tomando en cuenta la totalidad de los datos.
        2. cantidad_filmaciones_dia de la misma manera que la anterior, retorna la cantidad de películas estrenadas en el día de la semana proporcionado en el total del dataset.
        3. score_titulo mediante una petición a esta consulta, devuelve el titulo, el año de estreno y el escore de una filmación proporcionada
        4. votos_titulo esta consulta devuelve la cantidad de cotos y el valor promedio de las votaciones siempre y cuando esta supere las 2000 valoraciones.
        5. get_actor esta consulta retorna el éxito de un actor, cantidad de filmaciones, cantidad de películas donde actuó, retorno total, el retorno promedio y las peliculas en las que participo.
        6. get_director devuelve información de un director como las películas que ha dirigido y ganancias de las filmaciones.
        7. get_recomendacion esta función devuelve 5 películas recomendadas en función de la elegida
> [!IMPORTANT]
> La ejecución de este archivo sólo se lleva a cabo en Render, aunado a ello, se puede visualizar el código comentado en el archivo ``recomendationsystem.ipynb``para entender su funcionamiento. Importante mencionar que solo está detallado el sistema de recomendación y porque se eligió ese modelo. 👀
### ✔️```Resultados```
![FastAPI](/assets/FastAPI.png)


5. ## 🖥️ Virtualización y deployment
Para la ejecución de las API´s se recurrió al alojamiento del proyecto en los repositorios de GIt Hub, conectado el repositorio con Render, se realizó el deployment, esto con la finalidad de poder visualizar las consultas y así tener la información disponible para consumo. 
> [!IMPORTANT]
> Cabe recalcar que se utilizó render en modo gratuito por lo que se redujeron los datos para su correcto funcionamiento.
![GitHub](/assets/GitHub.png)




![GitHub](/assets/Recomendation.png)



# ✔️```Herramientas y librerías utilizadas```

Para el desarrollo de este proyecto se manipularon las siguientes herramientas:

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%2370399F.svg?style=for-the-badge&logo=seaborn&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=ffffff)
![venv](https://img.shields.io/badge/Virtualenv-venv-%2300FFFF?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Render](https://img.shields.io/badge/Render-46E3B7.svg?style=for-the-badge&logo=Render&logoColor=white)


# ✔️```Replicación del proyecto```
El sistema de recomendación aquí mostrado, puedes replicarlo por tí mismo, así como corroborar código, realizar modificaciones, realizar consultas, ejecutar pruebas, como base de estudio e incluso mejorarlo. Para ello es necesario que sigas los siguientes pasos:
1. Forkea este repositorio.
2. Sincronizarlo en tu ordenador.
3. Configura tu entorno virtual.
4. Activa tu entorno virtual.
5. Asegurate de tener los archivos necesarios: ``main.py``, carpeta ``data`` con los datos en formato parquet.
6. Instala las dependencias especificadas en el archivo ``requirements.txt``.
7. Ejecuta el comando ```uvicorn main:app --reload``` para iniciar la aplicación.
8. Despliega el Render en tu ordenador accediendo a tu localhost y el puerto dedicado mismo que te será proporcionado cuando ejecutes el paso anterior.


# ✔️```Posibles mejoras del proyecto```
1. Ampliación del Dataset:
    - Incorporar más datos, como reseñas de usuarios y clasificaciones detalladas, podría mejorar la precisión de las recomendaciones y la calidad de la información proporcionada.

2. Mejora de Algoritmos de Recomendación:
    - Explorar otros algoritmos de recomendación como modelos basados en redes neuronales o recomendaciones colaborativas podría ofrecer recomendaciones aún más precisas y personalizadas.

3. Interfaz de Usuario:
    - Desarrollar una interfaz de usuario interactiva y amigable podría mejorar significativamente la experiencia del usuario final, haciendo la API más accesible y atractiva.


# ✔️```Conclusión```

Para dar cierre al desarrollo del proyecto, se concluye que el proyecto ha logrado crear una API robusta y eficiente para la consulta y recomendación de películas, utilizando técnicas avanzadas de procesamiento de texto y algoritmos de similitud. Con una base sólida establecida, existen múltiples oportunidades para expandir y mejorar el sistema, lo que podría llevar a un servicio de recomendación de películas altamente competitivo y útil para los usuarios e incluso el uso en plataformas de streaming.



# ✔️```Sobre mi```

| [<img src="https://media.licdn.com/dms/image/D5603AQEnBacsLH1pnA/profile-displayphoto-shrink_800_800/0/1715214794765?e=1727913600&v=beta&t=2UKwQG8Hd4qK4Ac_R40acaT1UojfqqtOkECmPSpxs4s" width=150><br><sub>Juan DIego Hernández Camacho</sub>](https://www.linkedin.com/in/juan-diego-hernandez-camacho-5176022aa//)


Gracias por leer este repositorio 💛