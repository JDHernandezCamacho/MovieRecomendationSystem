# Librerias utilizadas
import pandas as pd
from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# -------------------------------------------------------------------------------------------------------------------------------------------------- #
                                                ### INSTANCIAS 
# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# Instanciacion del objeto FastAPI
app = FastAPI()

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
                                                ### VARIABLES
# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# Carga los datos desde un archivo parquet 
path_data = 'Data/dataset_movies.parquet'       # Ruta del archivo
df = pd.read_parquet(path=path_data)            # DataFrame con los datos cargados

# Carga el dataset de películas para Recomendacion
df_Ml = pd.read_parquet('Data/dataset_movies_to_ML.parquet')

# Diccionario de los meses en español con su respectivo numero identificador
meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5,
    'junio': 6, 'julio': 7, 'agosto': 8, 'septiembre': 9,
    'octubre': 10, 'noviembre': 11, 'diciembre': 12
}
# Diccionario de los días de la semana en español con su identificador
dias = {
    'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 'viernes': 4,
    'sabado': 5, 'domingo': 6
}

# Vectorización
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_Ml['overview'])

# Calcular similitud del coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Crear un índice para mapear los títulos a los índices, y nos aseguramos que no existan duplicados
indices = pd.Series(df_Ml.index, index=df_Ml['title']).drop_duplicates()


# -------------------------------------------------------------------------------------------------------------------------------------------------- #
                                                ### FUNCION DE APOYO
# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# Función de apoyo para buscar actores en las columas
def buscar_actor(nombre_actor):
    """
    Esta función crea una columna llamada count en la cual se hace un conteo si aparece el actor en ella
    Parametros: 
        nombre_actor: el nombre del actor a buscar
    Retorno: 
        total_count: suma de las veces que aprecio el actor en todos los datos
        films: dataframe con las películas donde aparecio el actor
        retorno: suma el retorno de lo recaudado en todos las peliculas
        promedio: promedio de recaudado por película
    """
    nombre_actor = nombre_actor.lower()
    # Contar el número de veces que aparece el actor en cada fila
    df['count'] = df['actor_names'].apply(lambda actors: sum(actor.lower() == nombre_actor for actor in actors))
    # variable que guarda la suma de todos los conteos
    total_count = df['count'].sum()
    
    # Mascara que busca y filtra la columna count cuando es diferente de cero, es decir, aparece el actor
    filtro = df[df['count'] != 0]
    # variable que contiene las peliculas en las que ha participado el actor
    
    # Mascara que busca y filtra la columna count cuando es diferente de cero, es decir, aparece el actor
    films = filtro['title']
    # Variable que contiene la suma total del retorno del autor 
    retorno = filtro['return'].sum()
    # Calculo del promedio de retorno por película
    promedio = retorno / total_count
    # Variables a retornar 
    return total_count, filtro, retorno, round(promedio,2)

# Función de apoyo para buscar directores en las columas
def buscar_director(nombre_director):
    """
    Esta función corrobora si el nombre de un director recibido por parámetro se encuentra en el dataset,
    crea una columna llamada count por pelucila si aparece el director en ella
    filtra las peliculas, con la fecha de lanzamiento 
    Parametros: 
        nombre_director: el nombre del  director a buscar
    Retorno: 
        total_count: suma de las veces que aprecio el actor en todos los datos
    """
    nombre_director = nombre_director.lower()
    # Contar el número de veces que aparece el actor en cada fila
    df['count'] = df['director'].apply(lambda directors: sum(director.lower() == nombre_director for director in directors))
    # variable que guarda la suma de todos los conteos
    total_count = df['count'].sum()
    
    # Mascara que busca y filtra la columna count cuando es diferente de cero, es decir, aparece el director
    filtro = df[df['count'] != 0]
    # variable que contiene las peliculas en las que ha participado el actor con año de lnazamiento, retorno, ganancia y retorno
    # Mascara que busca y filtra las columnas cuando count es diferente de cero, es decir, aparece el director
    films = filtro[['title','release_year','return','budget', 'revenue']]
    # Variable que contiene la suma total del retorno del autor 
    retorno = filtro['return'].sum()
    # Calculo del promedio de retorno por película
    promedio = retorno / total_count
    # Variables a retornar 
    return total_count, films, retorno, round(promedio,2)

# Funvión de apoyo para buscar y ordenas las peliculas proporcionadas en un DataFrame
def listar_peliculas(films):
    """
        Esta función recibe un dataframe de peliculas filtrado y obtiene la informacion de 
        titulo, año de lanzamiento, retorno, ganancia y presupuesto por cada película
            Parametros: 
                film: dataframe cde peliculas filtrado
            Retorno:
                lista_films: lista de diccionarios con la informacion especificada
    """
    # se crea una lista vacia
    lista_films = []
    # Ciclo que recorre cada una de las peliculas dentro del dataframe
    for i in range(len(films)):
        # se crea el diccionario con la informacion detallada
        movie = {
            'titulo': str(films.iloc[i]['title']),
            'año': str(films.iloc[i]['release_year']),
            'retorno_pelicula': str(films.iloc[i]['return']),
            'ganancia': str(films.iloc[i]['revenue']),
            'presupuesto': str(films.iloc[i]['budget'])
        }
        # se agrega cada pelicula con su informacion a la lista vacia creada antes del ciclo for
        lista_films.append(movie)
    # retorna la lista creada
    return lista_films

# Función que realiza la recomendación
def recomendar_peliculas(dato, cosine_sim = cosine_sim):
    """
        Función que realiza la recomendación de peliculas 
        Parametros: 
            dato = Recibe el id de la posición de la película 
            cosine_sim = la variable de la similitud del coseno ya cargada
        Retorna:
                movie_indices = Los índices de las películas recomendadas
    """
    posicion = dato 
    sim_scores = list(enumerate(cosine_sim[posicion]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Recomendamos las 5 películas similares
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
                                                    ### FUNCIONES API
# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# Función que calcula la cantidad de filmaciones estrenadas por mes
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    """ 
        Esta función calcula la cantidad de peliculas estrenadas por un mes n específico
        recibe como parámetro el nombre de un mes en español y retorna la cantidad
        peliculas estrenadas en ese mes junto con una cadena informativa.
        Parametros: 
            mes: mes en español y formato string
        Retorno:
            cantidad: Catidad de peliculas estrenadas ese mes
    """
    mes = mes.lower()
    # Condicional que pregunta si mes se encuentra en el diccionario meses
    if mes in meses:
        # Variable con el numero de mes
        month_number = meses[mes]
        # Nos aseguramos de que la columna de fechas en el dataframe sea de tipo datetime
        # En caso de que exista algún nulo, se ignora con errors='corce'
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        # Máscara para filtra por mes, específicamente en formato numérico
        cantidad = df[df['release_date'].dt.month == month_number].shape[0]
        # Retorno del resultado de la función 
        return f"{cantidad} películas fueron estrenadas en el mes de {mes.capitalize()}"
    # condicional en caso de que el mes no se encuentre en el diccionario
    else:
        # Retrorno en caso de que no exista el mes
        return f"{mes.capitalize()} no es un mes válido. Por favor ingrese un mes en español."


# Función que calcula la cantidad de filmaciones estrenadas por dia
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    """ 
        Esta función calcula la cantidad de peliculas estrenadas por un día en específico
        recibe como parámetro el nombre de un día en español y retorna la cantidad
        peliculas estrenadas en ese día junto con una cadena informativa.
        Parametros: 
            dia: mes en español y formato string
        Retorno:
            cantidad: Catidad de peliculas estrenadas en ese dia
    """
    # Convierte el strind a minúsculas
    dia = dia.lower()
    # condicional en caso de que dia se encuentre enel diccionario dias
    if dia in dias:
        # Consulta en el diccionario el número del día proporcionado 
        day_number = dias[dia]
        # # Nos aseguramos de que la columna release_date en el dataframe sea de tipo datetime
        df['release_date'] = pd.to_datetime(df['release_date'])
        # Filtrar por día en formato numérico
        cantidad = df[df['release_date'].dt.day_of_week == day_number].shape[0]
        # Retorna el resultado de la función 
        return f"{cantidad} películas fueron estrenadas en día {dia.capitalize()}"
    # condicional en caso de que dia no sea un dia de la semana
    else:
        # Resultado cuando el día proporcionado no exista
        return f"{dia.capitalize()} no es un día válido. Por favor ingrese un día en español."


# Función que retorna detalles (titulo, año de estreno y popularidad) de una pelicula
@app.get('/score_titulo/{titulo_de_la_filmacion}')
def score_titulo(titulo_de_la_filmacion: str):
    """
        Esta función recibe por parámetro un titulo de una película, busca y filtra el nombre en la columna title,
        y retorna el primer valor mostrado titulo, año de lanzamiento y popularidad).
        Parametros: 
            titulo_de_la_filmacion = Titulo de una película
        Retorno:
            string = Titulo de una película, año de estreno y popularidad.

    """
    # Mascara que busca y filtra la película por título (en minusculas para que coincida)
    film = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    # Condicional pregunta si existe un valor en la variable film, es decir si
    # existen peliculas con el nombre recibido como parámetro 
    if not film.empty:
        # variable que guarda el primer titulo de pelicula con esa coincidencia
        titulo = film['title'].values[0]
        # variable que extrae en año de lanzamiento de la película encotrada
        anio_estreno = film['release_year'].values[0]
        # Variale con la popularidad o score de la pelicula encontrada
        score = film['popularity'].values[0]
        # retorno de la función
        return f"La película {titulo} fue estrenada en el año {anio_estreno} con un score/popularidad de {round(score,2)}"
    # Condicional, en caso de que la pelicula no exista en los datos
    else:
        # Error o excepción
        detail = f"Película {titulo_de_la_filmacion.capitalize()} no fue encontrada, por favor revisa tu ortografía" 
        raise HTTPException(status_code=404, detail=detail)


# Función que retorna detalles (titulo, año de estreno y popularidad) de una pelicula
@app.get('/votos_titulo/{titulo_de_la_filmacion}')
def votos_titulo(titulo_de_la_filmacion: str):
    """
        Esta función recibe por parámetro un titulo de una película, busca y filtra el nombre en la columna title,
        y retorna datos como el titulo, la cantidad de votos y el valor promedio de votaciones si tiene mas de 
        2k votaciones
        Parametros: 
            titulo_de_la_filmacion = Titulo de una película
        Retorno:
            string = Titulo de una película, año de estreno y popularidad.

    """
    # Mascara que busca y filtra la película por título (en minusculas para que coincida)
    film = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    # Condicional pregunta si existe un valor en la variable film, es decir si
    # existen peliculas con el nombre recibido como parámetro 
    if not film.empty:
        # variable que guarda el primer titulo de pelicula con esa coincidencia
        titulo = film['title'].values[0]
        # variable que extrae en año de lanzamiento de la película encotrada
        anio_estreno = film['release_year'].values[0]
        # Variale con el conteo de votos de la pelicula encontrada
        vote_count = film['vote_count'].values[0]
        # Variale con la popularidad o score de la pelicula encontrada
        vote_average = film['vote_average'].values[0]
        # condicional si el conteo de votos sobrepasa los 2k 
        if int(vote_count) > 2000:
            # Retorno de la función con los detalles obtenidos
            string = f"La película {titulo} fue estrenada en el año {anio_estreno} cuenta con un total de {vote_count} valoraciones y un promedio de {vote_average}"
            return string
        # Condicional en caso de que la pelicula no tenga mas de 2K valoraciones
        else:
            string = f"La película {titulo} no cuenta con las valoraciones necesarias."
            return string
    # Condicional, en caso de que la pelicula no exista en los datos
    else:
        # Error o excepción
        detail = f"Película {titulo_de_la_filmacion.capitalize()} no fue encontrada, por favor revisa tu ortografía" 
        raise HTTPException(status_code=404, detail=detail)


# Función que retorna detalles (titulo, año de estreno y popularidad) de una pelicula
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    """
        Esta función recibe por parámetro el nombre de un actor, busca y filtra el nombre en la columna actor_names,
        y retorna datos como el nombre del actor, la cantidad de filmaciones donde ha participado y el valor promedio 
        de retorno.
        Parametros: 
            nombre_actor = nombre del actor 
        Retorno:
            string = Nombre del actor, peliculas de participación y el promedio de retorno.

    """
    # Se convierte el string a minusculas
    nombre_actor = nombre_actor.lower()
    # Variable que contiene el total de veces que aparece el actor, 
    count, films, retorno, promedio = buscar_actor(nombre_actor)
    
    # Condicional pregunta si existe un valor en la variable count, es mayor a cero,
    # entonces si existen peliculas con el nombre de actor recibido como parámetro 
    if not films.empty:
        # Se manda el dataframe a la función para listar las peliculas
        lista_pelis = listar_peliculas(films)
        # Variable con la informacion a retornar
        dictionary = {
            'nombre_actor': nombre_actor,
            'cantidad_peliculas': int(count),
            'retorno_total': round(retorno,2),
            'promedio_por_pelicula': round(promedio,2),
            'detalle_de_peliculas': lista_pelis
        }
        string = f"{nombre_actor.capitalize()}, ha actuado en {count} películas, recibiendo {round(retorno,2)} de retorno consiguiendo un promedio de {promedio} por película. Lista de peliculas \n {films.to_string(index= False)}"
        # retorna la informacion del string
        return dictionary
    # En caso de que el df de films este vacio, retorna mensaje de no encontrado
    else:
        raise HTTPException(status_code=404, detail=f"Actor {nombre_actor.capitalize()} no encontrado en las Base de datos")


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    """
        Esta función recibe por parámetro el nombre de un actor, busca y filtra el nombre en la columna actor_names,
        y retorna datos como el nombre del actor, la cantidad de filmaciones donde ha participado y el valor promedio 
        de retorno.
        Parametros: 
            nombre_actor = nombre del actor 
        Retorno:
            string = Nombre del actor, peliculas de participación y el promedio de retorno.

    """
    # Se convierte el string a minusculas
    nombre_director = nombre_director.lower()
    # Variables que contienen el total de veces que aparece el director, df de peliculas donde aparece
    # el retorno toal y el promedo por pelicula 
    count, films, retorno, promedio = buscar_director(nombre_director)
    # Condicional pregunta si la variable films no esta vacía, en caso verdadero,
    # existen peliculas con el nombre de director recibido como parámetro 
    if not films.empty:
        movies = listar_peliculas(films)
        # string = f'El director {nombre_director} dirigió {count} películas tiene retorno total de {retorno} y un promedio {promedio}, estas son las peliculas {type(movies)} '
        #Pasando la información a un diccionario 
        dictionary = {
            'director': nombre_director.capitalize(),
            'peliculas_dirigidas': int(count),
            'retorno_total': round(retorno,2),
            'promedio_retorno': promedio,
            'detalle_peliculas': movies
        }
        # Retorno del diccionario 
        return dictionary


    # # En caso de que el df de films este vacio, retorna mensaje de no encontrado
    else:
        raise HTTPException(status_code=404, detail=f"Director {nombre_director.capitalize()} no encontrado en las Base de datos")



# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Funciñon de recomendación 

@app.get('/recomendacion/{nombre_pelicula}')
def recomendacion(nombre_pelicula: str):
    id = -1
    nombre_pelicula = nombre_pelicula.lower()
    for i, ind in enumerate (indices):
        if nombre_pelicula == indices.index[i].lower():
            id = i
            break
    if id < 0:
        raise HTTPException(status_code=404, detail=f"Película {nombre_pelicula.capitalize()} no encontrada en las Base de datos")
    else:
        indices_m = recomendar_peliculas(id)
        lista_peliculas = []
        for ind in indices_m:
            movie_rec = df_Ml['title'].iloc[ind]
            lista_peliculas.append(movie_rec)
        dictionary = {
            'pelicula': nombre_pelicula.capitalize(),
            'id': id,
            'recomendacion_peliculas': lista_peliculas
        }
        
    return dictionary


 

# Función de la direccion raíz del proyecto
@app.get("/")
def reload_root():
    
    return {f"Bienvenido a MovMod, hay {df['release_date'].count()} datos cargados para consultas y {df_Ml['title'].count()} para recomendación."}