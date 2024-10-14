# Clasificador de imágenes de animales

Este proyecto utiliza técnicas de Deep Learning para clasificar imágenes de diferentes especies de animales utilizando una red neuronal convolucional basada en ResNet50. 
El modelo ha sido entrenado con un conjunto de datos que incluye cinco clases de animales.

## Descripción

El objetivo de este proyecto es clasificar imágenes en cinco categorías de animales: 
- Perro
- Caballo
- Elefante
- Mariposa
- Gallina

El modelo ha sido desarrollado utilizando TensorFlow y Keras, aprovechando la arquitectura ResNet50 para mejorar la precisión en la clasificación.

## Características

- Clasificación de imágenes con un alto grado de precisión.
- Uso de técnicas de aumentación de datos para mejorar la generalización del modelo.
- Implementación de estrategias de regularización para evitar el sobreajuste.
- API REST para realizar predicciones sobre nuevas imágenes.

## Tecnologías Utilizadas

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Flask (para la API REST)
- Git (para control de versiones)

## Archivos del Proyecto

- `app.py`: Implementa una API REST utilizando Flask para permitir la clasificación de nuevas imágenes a través de peticiones HTTP.
- `functions.py`: Contiene funciones auxiliares utilizadas para el procesamiento de imágenes y la carga del modelo.
- `modelo.py`: Define y entrena el modelo de clasificación. Contiene la arquitectura de la red neuronal y las funciones necesarias para la optimización y evaluación del modelo.
- `modelo.h5`: Archivo que contiene el modelo entrenado. Este archivo se genera después de completar el entrenamiento y se utiliza para realizar predicciones sobre nuevas imágenes. Puede ser cargado utilizando Keras para hacer inferencias sin necesidad de reentrenar el modelo.
- `styles.css`: Archivo CSS que contiene estilos para la interfaz de usuario. Se utiliza para dar formato y mejorar la apariencia de la aplicación web, asegurando que sea visualmente atractiva y fácil de usar.
- `home.html`: Archivo HTML que se utiliza para la interfaz de usuario del proyecto (si aplica).

## Instalación

1. Clona este repositorio:

   git clone (https://github.com/Lucosiar/Api-Clasificadora-de-animales)

2. Crear entorno virtual
   Windows: python -m venv venv
   Linux\Mac: python3 -m venv venv

4. Activar entorno virtual
   Windows: venv\Scripts\activate
   Linux\Mac: source venv/bin/activate
   
6. Instalar Requirements
   pip install -r requirements.txt
   
8. Ejecutar código
   python app.py




