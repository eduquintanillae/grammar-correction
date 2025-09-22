# Corrección Gramatical Automática

## Descripción del Desafío

Tu objetivo es construir un sistema de **corrección gramatical automática (Grammatical Error Correction, o GEC, en inglés)** para textos en inglés.

Para ello, deberás comparar dos enfoques:

1. Un modelo grande de lenguaje off-the-shelf (por ejemplo, ChatGPT o Claude), utilizando técnicas de **prompt engineering**.
2. Un modelo más pequeño (e.g., alguno de la serie T5 de Google) entrenado específicamente para esta tarea mediante **fine-tuning**.

Queremos que analices y compares la calidad de los resultados generados por ambos enfoques y extraigas conclusiones sobre su rendimiento.

Se te proporciona un **dataset adicional en formato CSV con textos en inglés en un contexto médico**. Este conjunto de datos incluye frases con errores gramaticales reales dentro de este contexto. Deberás:

- Evaluar el desempeño de ambos enfoques sobre este conjunto específico.
- Analizar las posibles dificultades que surgen al corregir textos técnicos.
- Proponer e implementar mejoras si el rendimiento del modelo en este dominio es deficiente.

## Dataset

Deberás utilizar al menos uno de los siguientes corpus públicos de corrección gramatical en inglés:

- **JFLEG** – [https://jfleg.github.io/](https://jfleg.github.io/)  
  Contiene frases con errores y múltiples correcciones escritas por humanos.

- **BEA 2019 Shared Task** – [https://www.cl.cam.ac.uk/research/nl/bea2019st/](https://www.cl.cam.ac.uk/research/nl/bea2019st/)  
  Incluye conjuntos como FCE, W&I+LOCNESS, entre otros.

Además, se incluye un **dataset adicional en formato CSV** con frases en inglés en contexto **médico**. Este conjunto debe ser considerado para evaluación y mejora del sistema.

## Preparación y exploración del dataset

Antes de entrenar cualquier modelo, deberás:

- Cargar y explorar el o los datasets elegidos. GEC utiliza el formato M2 para anotar las etiquetas, así que deberás buscar o implementar alguna forma para parsear este formato.
- Identificar las características principales del corpus general y del dataset médico.
- De ser necesario, implementar un paso de **preprocesamiento** que convierta las muestras del dataset a un formato adecuado para los modelos que usarás.

Explica en tu entrega las decisiones tomadas durante esta etapa.

## Modelado

### Enfoque 1: Modelo de lenguaje con prompt engineering

Usa un modelo de lenguaje grande, como ChatGPT o Claude, para corregir las oraciones con errores usando instrucciones ("prompts").

### Enfoque 2: Fine-tuning de T5-small

Entrena un modelo en el dataset preprocesado. Este modelo debe tomar oraciones incorrectas como entrada y generar oraciones corregidas como salida. Si necesitas cómputo en GPU, puedes utilizar Google Colab.

## Método de evaluación

Explica claramente **qué métrica(s) eliges** para comparar los resultados (puedes usar más de una si lo consideras necesario) y **por qué** son apropiadas para esta tarea.

Incluye ejemplos concretos en tu análisis: oraciones de entrada, salidas corregidas por cada modelo y su calidad relativa.

Evalúa también el rendimiento sobre el dataset médico. Si encuentras que los modelos tienen dificultades con este tipo de texto, describe los errores más comunes y propone formas de mejorar el desempeño.

Bonus: comparar tiempos de respuesta de ambos enfoques.

## Entregables esperados

Tu entrega debe incluir:

- Código o cuaderno (Jupyter/Colab) con tu implementación.
- Un resumen de tus observaciones y análisis.
- Justificación de las métricas usadas.
- Comparación de outputs con ejemplos.
- Conclusiones sobre ventajas y limitaciones de ambos enfoques.
- Evaluación específica sobre el dataset médico y mejoras propuestas si es necesario.

## Notas

- No uses datos privados ni confidenciales.
- Se valorará tu capacidad de análisis, claridad en la presentación y buenas prácticas de programación.
- Puedes usar bibliotecas como HuggingFace Transformers, pero evita usar soluciones ya hechas específicamente para GEC. Recuerda que queremos evaluar cómo te enfrentas al reto de fine-tunear un modelo.
