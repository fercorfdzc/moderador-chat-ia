
# Moderador Automático de Conversaciones Escritas en Vivo
Este proyecto consiste en el desarrollo de un sistema inteligente capaz de detectar lenguaje tóxico y discurso de odio en chats en tiempo real, utilizando técnicas modernas de Procesamiento de Lenguaje Natural (NLP).

El sistema analiza mensajes escritos por los usuarios y los clasifica en distintas categorías como:


## Caracteristicas
- Contenido Limpio 
- Lenguaje  toxico 
- Disscurso de odio 

## Objetivo General
Desarrollar un sistema de moderación automática para chats en tiempo real que identifique y clasifique contenido ofensivo mediante inteligencia artificial, permitiendo su integración en entornos reales de comunicación digital.

###  Objetivos Espeificos
- Diseñar un pipeline de NLP para análisis de mensajes cortos.  
- Implementar fine-tuning de un modelo para el idioma Español/Ingles.
- Crear un dataset etiquetado (clean, toxic, hate).                 
- Desarrollar una API REST para el modelo. 
- Integrar un bot para chats en vivo. 
- Definir reglas de moderación basadas en probabilidades
- Evaluar el modelo con métricas (precision, recall, F1).           

## ¿Qué problema resuelve? 
los chats en vivo suelen contener_
- Insultos 
- Acoso 
- Lenguaje descriminatorio 

la Modelacion Manual:
- No es escalable 
- Es lenta 
- No cubre todos los mensajes

Este sistema automatiza la detección para mejorar la seguridad digital.

## Relevancia
- Mejora la convivencia en comunidades digitales
- Reduce carga de moderadores humanos
- Apoya a plataformas de streaming y creadores de contenido
- Promueve entornos más seguros
  
## Tecnologias Utilizadas

| Área | Herramientas | 
|-----------|-----------|
| Lenguaje    | Python    | 
| Deep Learning  | PyTorch   | 
| NLP    | Hugging Face Transformers   | 
| API | FastAPI   | 
| Datos  | Pandas, Conjuntos de datos  | 
| Evaluación | Scikit-learn   | 
| Versiones   | Git + GitHub | 

  
## Arquitectura del Sistema
- Usuario envía mensaje en chat 
- Bot captura el mensaje 
- API procesa el texto 
- Modelo NLP clasifica el contenido
- Sistema aplica reglas de moderación
  
## Datos y Recursos
1. Recolección
- Datasets Personalizados
2. Preprocesamiento
- Minúsculas
- Limpieza de texto
- Manejo de emojis
- Tokenización
3. Modelado
- Transformer preentrenado
- Red neuronal de clasificación
- Clasificación multiclase
4. Evaluación
- Precision
- Recall
- F1-score
- Matriz de confusión
  
## Metricas de Evaluación
- Precision: Exactitud en predicciones positivas
- Recall: Capacidad de detectar contenido ofensivo
- F1-score: Balance entre precision y recall
- Matriz de confusión: Análisis de errores
  
## Aprendizaje del Modelo
- Aprendizaje supervisado
- Datos etiquetados
- Ajuste de hiperparámetros
- Mejora iterativa mediante retroalimentación
  
## Retroalimentación
El sistema mejora mediante:
- Análisis de errores
- Ajuste de umbrales
- Reentrenamiento del modelo
- Refinamiento del dataset
  
## Desafíos
- Desbalance de datos
- Ambigüedad del lenguaje
- Sarcasmo e ironía
- Falsos positivos
- Limitaciones computacionales
  
## Aplicaciones
- Moderación automática de chats
- Plataformas de streaming
- Redes sociales
- Sistemas de filtrado de contenido
  
## Impacto Esperado
- Mejora de entornos digitales 
- Reducción de contenido dañino 
- Apoyo a moderadores humanos 
- Uso responsable de IA
  
## Futuras Mejoras
- Optimización con ONNX o cuantización
- Panel web de estadísticas
- Soporte multilenguaje
- Detección de emociones (extensión con ML)
  
## Authors

- [Jonathan Fernandez Cordova ](https://github.com/fercorfdzc)
- [Renata Castro Olmos ](https://github.com/RenataCastro19)
- [Brayan Emanuel Vazques Peña ](https://github.com/Brayan1224)
- [Adriana de los Angeles Toto Chapol ](https://github.com/AdrianaTc26)
