import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

archivos = ["ex1data2(Home_1f)","petrol_consumption", "oceano_simple"]

def leer_datos(archivo):
    return np.genfromtxt(archivo+".csv", delimiter='\t', skip_header=1)

def dividir_X_y(datos):
    X = datos[:,:-1]
    y = datos[:,-1:]
    return X, y

def normalizar_datos(datos):
    media = datos.mean(axis=0)
    desv_est = datos.std(axis=0)
    datos = (datos - media)/ desv_est
    return datos, media, desv_est

def crear_entrenamiento_prueba(datos):
    n = int(datos.shape[0] * 0.7)
    entrenamiento, prueba = datos[:n,:], datos[n:,:]
    return entrenamiento, prueba

def calcular_costo(X, y, t):
    costo = np.matmul(X,t)
    costo = np.square(costo-y)
    costo = np.sum(costo)
    costo = costo / (2 * y.shape[0])
    return costo

def gradiente_descendiente(X, y, t, num_it, tasa_apren):
    costos = np.zeros(num_it)
    for i in range(num_it):
        aux = np.matmul(X, t) - y
        aux = np.matmul(np.transpose(X), aux)
        aux = np.divide(aux, y.shape[0])
        aux = np.multiply(aux, tasa_apren)
        t = t - aux
        costos[i] = calcular_costo(X, y, t)
    return t, costos

def ecuacion_normal(X, y):
    X_t = np.transpose(X)
    return np.matmul(np.linalg.inv(np.matmul(X_t, X)), np.matmul(X_t, y))

def Experimento1():
    print("Experimento 1:\n")
    tabla_costos = []
    for archivo in archivos:
        data = leer_datos(archivo)
        data = normalizar_datos(data)[0]
        X, y = dividir_X_y(data)
        X = np.c_[np.ones(X.shape[0]),X]

        X_train, X_test = crear_entrenamiento_prueba(X)
        y_train, y_test = crear_entrenamiento_prueba(y)

        y_train = np.reshape(y_train, y_train.shape[0])
        y_test = np.reshape(y_test, y_test.shape[0])

        t = ecuacion_normal(X_train, y_train)

        costos_entrenamiento = calcular_costo(X_train, y_train, t)
        costos_prueba = calcular_costo(X_test, y_test, t)

        tabla_costos.append([costos_entrenamiento,costos_prueba])

    tabla_costos = pd.DataFrame(tabla_costos,columns=["Costos de entrenamiento", "Costos de prueba"],index=archivos)
    print(tabla_costos)

def Experimento2():
    print("Experimento 2:")
    n_it = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    tasas_aprendizaje = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    for archivo in archivos:
        data = leer_datos(archivo)
        data = normalizar_datos(data)[0]
        X, y = dividir_X_y(data)
        X = np.c_[np.ones(X.shape[0]),X]

        X_train, X_test = crear_entrenamiento_prueba(X)

        y_train, y_test = crear_entrenamiento_prueba(y)
        y_train = np.reshape(y_train, y_train.shape[0])
        y_test = np.reshape(y_test, y_test.shape[0])

        print("\nArchivo: ", archivo)
        tabla_costos = []
        for it in n_it:
            costos = []
            for tasa in tasas_aprendizaje:
                t = np.zeros(X.shape[1])
                t = gradiente_descendiente(X_train, y_train, t, it, tasa)[0]
                costo_train = calcular_costo(X_train,y_train,t)
                costos.append(costo_train)
            tabla_costos.append(costos)
        tabla_costos = pd.DataFrame(tabla_costos, columns=tasas_aprendizaje, index=n_it)
        #tabla_costos.rename_axis('it').rename_axis('tasa', axis='columns')
        print(tabla_costos)

def Experimento3():
    print("Experimento 3:\n")
    data = leer_datos("ex1data2(Home_1f)")
    data_train = crear_entrenamiento_prueba(data)[0]
    data_train = normalizar_datos(data_train)[0]
    X, y = dividir_X_y(data_train)
    X_bias = np.c_[np.ones(X.shape[0]), X]
    y = np.reshape(y, y.shape[0])

    t_en = ecuacion_normal(X_bias,y)
    t_en = np.reshape(t_en,t_en.shape[0])
    t_en = np.flip(t_en)
    plt.subplot(1, 2, 1)
    plt.title("Regresion Lineal con Ecuacion Normal")
    plt.plot(X, y,'ro')
    polinomio1 = np.poly1d(t_en)
    plt.plot(X,polinomio1(X))

    t_gd = np.zeros(X_bias.shape[1])
    t_gd = gradiente_descendiente(X_bias,y,t_gd,500,0.1)[0]
    t_gd = np.flip(t_gd)
    plt.subplot(1, 2, 2)
    plt.title("Regresion Lineal con Gradiente Descendiente")
    plt.plot(X, y, 'ro')
    polinomio2 = np.poly1d(t_gd)
    plt.plot(X, polinomio2(X))
    plt.show()

def Experimento4():
    print("Experimento 4:\n")
    for archivo in archivos:
        data = leer_datos(archivo)
        data_train, data_test = crear_entrenamiento_prueba(data)
        data_train = normalizar_datos(data_train)[0]
        X, y = dividir_X_y(data_train)
        X = np.c_[np.ones(X.shape[0]), X]
        y = np.reshape(y, y.shape[0])

        t = np.zeros(X.shape[1])
        costos = gradiente_descendiente(X, y, t, 500, 0.1)[1]
        print(costos)
        plt.title(archivo)
        plt.plot(costos)
        plt.show()
    print("\n")

#Experimento1()
#Experimento2()
#Experimento3()
#Experimento4()