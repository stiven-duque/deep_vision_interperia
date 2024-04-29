
# Proyecto de Procesamiento de Imágenes y Pruebas de Modelos (Deep vision INTERPERIA)

Este proyecto proporciona funciones para el procesamiento de imágenes, pruebas de modelos de aprendizaje profundo e implementaciones modificadas del modelo VGG16 para la detección de armas de fuego y cuchillos.

## Funciones Principales

### `image_processing.py`

Este módulo contiene funciones para dividir imágenes en partes más pequeñas, calcular la entropía y la complejidad de las imágenes, y descartar imágenes basadas en umbrales de entropía y complejidad.

- `split_image(file, output_directory, split_width, overlap_percentage)`: Divide una imagen en partes más pequeñas y las guarda en un directorio de salida.
- `calculate_entropy(image)`: Calcula la entropía de una imagen.
- `calculate_complexity(image)`: Calcula el índice de complejidad de una imagen.
- `discard_images(dataset_path, entropy_threshold, complexity_threshold)`: Descarta imágenes que caen por debajo de los umbrales de entropía y complejidad.

### `test_model.py`

Este módulo contiene una función para probar un modelo de aprendizaje profundo y dibujar bounding boxes en una imagen original.

- `test_model(model, image_directory, image_original_path, output_directory, split_width, classes)`: Prueba un modelo en un conjunto de imágenes, dibuja bounding boxes en una imagen original y guarda la imagen resultante en un directorio de salida.

### `vgg_model.py`

Este módulo contiene una implementación modificada del modelo VGG16 para la detección de armas de fuego.

- `ModifiedVGG16Model`: Clase que implementa el modelo VGG16 modificado.

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/tu_usuario/tu_proyecto.git
```
2. Instala los requisitos:
```bash
conda install torch
```

## Uso 

1. Ejecuta el archivo `main.py`` para dividir las imágenes, descartar imágenes no deseadas y probar el modelo.


## Créditos 
- Autor: Fabian Stiven Duque Duque
- Contacto: fabian.duque@udea.edu.co