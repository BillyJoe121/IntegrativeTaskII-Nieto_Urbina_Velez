### Prompts usados con Gemini CLI

---

1.  **Análisis de EDA y Selección de Características:**
    "Basado en el análisis exploratorio de datos (EDA) realizado en `EDA.ipynb`, explica por qué la longitud de las oraciones fue determinada como un predictor débil para el sentimiento. ¿Qué implica la distribución casi idéntica de longitudes de oraciones para los sentimientos positivos y negativos?"

2.  **Comparación de Modelos (DNN vs. LSTM):**
    "El notebook `03_dense_rnn_lstm.ipynb` muestra que la Red Neuronal Densa (DNN) con TF-IDF obtuvo un rendimiento ligeramente superior al modelo LSTM, que es más complejo. ¿Cuáles podrían ser algunas razones potenciales para este resultado, considerando las características del conjunto de datos y los pasos de preprocesamiento?"

3.  **Uso de Capas de Regularización (SpatialDropout1D):**
    "En los modelos RNN y LSTM, se utiliza la capa `SpatialDropout1D`. Explica su función en comparación con una capa `Dropout` estándar y por qué es particularmente efectiva después de una capa de Embedding, haciendo referencia a las definiciones del modelo en `03_dense_rnn_lstm.ipynb`."

4.  **Conceptos de Máquinas de Turing y Redes Neuronales:**
    "El módulo `evaluate.py` incluye una función para conectar arquitecturas de redes neuronales con conceptos de Máquinas de Turing. Basándote en los docstrings de ese archivo, ¿podrías desarrollar por qué un modelo LSTM se considera teóricamente Turing-completo, mientras que una red Densa feed-forward no lo es?"

5.  **Interpretación de Modelos (Pesos de la Capa Densa):**
    "La función `analyze_dense_weights` en `evaluate.py` se utiliza para la interpretabilidad del modelo. Explica la lógica detrás de esta función. ¿Cómo ayuda el análisis de los pesos de la primera capa densa a comprender qué palabras asocia el modelo con un sentimiento positivo o negativo?"

6.  **Próximos Pasos para la Mejora del Modelo:**
    "Considerando todo el proyecto, desde la EDA hasta la evaluación del modelo, ¿cuál sería el siguiente paso más lógico para mejorar el rendimiento del modelo de análisis de sentimiento? ¿Debería el enfoque principal estar en la ingeniería de características, probar arquitecturas diferentes, o una aumentación de datos más agresiva? Justifica tu recomendación."
