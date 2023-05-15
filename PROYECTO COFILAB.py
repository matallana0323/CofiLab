# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:39:35 2023

@author: Juan Pablo
"""

## IMPORTAR LIBRERIAS 

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import requests
        import json
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split


## IMPORTAR BASES DE DATOS DE ALPHAVANTAGE

        base_url = "https://www.alphavantage.co/query"
        api_key = "FO2I3O6M59HSXESC"
        
        symbol = "AAPL"
        interval = "1min"
        start_time = "2022-06-01 09:30:00"
        end_time = "2023-04-30 16:00:00"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": api_key,
            "outputsize": "full",
            "starttime": start_time,
            "endtime": end_time
            }

        response = requests.get(base_url, params=params)
        data = json.loads(response.text)["Time Series (1min)"]
        
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
    
        print(df)

## ANALISIS DE LOS PRIMEROS REGISTROS Y TIPO DE DATOS

        df.head()
        df.dtypes 

## ANALISIS DESCRIPTIVO DE LA SERIA DE TIEMPO 

        # estadísticas descriptivas
        print(df['4. close'].describe())
    
## PATRONES DE ESTABILIDAD

        from statsmodels.tsa.seasonal import seasonal_decompose

        # descomposición de la serie de tiempo
        result = seasonal_decompose(df['4. close'], model='multiplicative', period=30)
        result.plot()

        plt.show()


## GRAFICAR BASE DE DATOS

        plt.plot(df.index, df["4. close"])
        plt.xlabel("Fecha")
        plt.ylabel("Precio de cierre")
        plt.title("Precios de cierre de las acciones de Apple")
        # Ajustar las etiquetas del eje x
        plt.xticks(rotation=90)
        plt.show()
        
## ALGORITMO MEDIA MOVIL

        ###  Creamos las medias móviles modelo 1
        short_window = 9
        long_window = 14
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = df['4. close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = df['4. close'].rolling(window=long_window, min_periods=1, center=False).mean()
        
        # Generamos las señales de compra y venta
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)

        signals['positions'] = signals['signal'].diff()
        
        # Graficamos las medias móviles y las señales de compra/venta
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(111, ylabel='Precio en $')
        
        df['4. close'].plot(ax=ax1, color='black', lw=2.)
        signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
        
        # Señales de compra
        ax1.plot(signals.loc[signals.positions == 1.0].index, 
                 signals.short_mavg[signals.positions == 1.0],
                 '^', markersize=10, color='b')
        
        # Señales de venta
        ax1.plot(signals.loc[signals.positions == -1.0].index, 
                 signals.short_mavg[signals.positions == -1.0],
                 'v', markersize=10, color='r')
        
        plt.show()
        
      
## MODELO PREDICTIVO (BOSQUES ALEATORIOS)

        # Calcula las medias móviles de 30y 60 períodos
        df['SMA9'] = df['4. close'].rolling(window=9).mean()
        df['SMA14'] = df['4. close'].rolling(window=14).mean()
        
        # Elimina los valores faltantes
        df.dropna(inplace=True)
        
        # Crea una nueva columna con la señal de compra o venta
        df['Signal'] = np.where(df['SMA9'] > df['SMA14'], 1, -1)

        # Divide los datos en un conjunto de entrenamiento y un conjunto de prueba
        X_train, X_test, y_train, y_test = train_test_split(df[['SMA9', 'SMA14']], df['Signal'], test_size=0.3, random_state=42)

        # Crea un modelo de bosques aleatorios con 100 árboles de decisión
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Entrena el modelo con el conjunto de entrenamiento
        model.fit(X_train, y_train)
        
        # Calcula la precisión del modelo en el conjunto de prueba
        accuracy = model.score(X_test, y_test)
        print('Precisión del modelo: %.4f' % accuracy)

# Carga nuevos datos de precios de la acción en un DataFrame
#new_data = pd.read_csv('new_data.csv')



        # Calcula las medias móviles de 30 y 60 períodos para los nuevos datos
        df['SMA9'] = df['4. close'].rolling(window=30).mean()
        df['SMA14'] = df['4. close'].rolling(window=60).mean()

        # Elimina los valores faltantes
        df.dropna(inplace=True)

        # Utiliza el modelo entrenado para hacer predicciones sobre los nuevos datos
        X_new = df[['SMA9', 'SMA14']]
        y_new = model.predict(X_new)

        # Imprime las predicciones
        print(y_new)



