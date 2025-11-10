**Objetivos**  
**Unidad 3 – Máquinas de Turing y Aplicaciones**  
OE3.1 Definir formalmente una Máquina de Turing y cada uno de sus componentes.  
OE3.2 Construir Máquinas de Turing como reconocedoras de lenguajes y calculadoras de funciones usando un lenguaje de programación.   
OE3.3 Comprender los conceptos básicos de aprendizaje automático, en especial, el procesamiento de lenguaje natural, el aprendizaje supervisado y las redes neuronales.  
OE3.4 Aplicar los conceptos de redes neuronales recurrentes, LSTM y Redes Neuronales de Turing para resolver tareas de NLP usando un lenguaje de programación.  
OE3.5 Codificar y enumerar Máquinas de Turing con el fin de establecer un marco estandarizado y riguroso para describir la estructura, comportamiento y límites de los distintos dispositivos computacionales.  
OE3.6 Reconocer la importancia de lenguajes que no son RE, RE que no son Recursivos para representar problemas complejos, explorar la computabilidad y así proporcionar una base teórica para la teoría del lenguaje formal.

**Nivel de uso de IAG** 

| 3\.  Colaboración con IAG  | Los estudiantes pueden apoyarse en la IAG para completar tareas o desarrollar entregables asociados a la actividad, aprovechando las capacidades de estas herramientas para mejorar los productos. Asimismo, se espera que los estudiantes lleven un registro de sus interacciones con la IAG y estén en la capacidad de modificar los resultados generados, demostrando comprensión y dominio conceptual. Este registro debe contener tanto el contenido de autoría propia proporcionado a la IAG como los prompts empleados.  | Tareas integradoras 1 y 2\.Estas tareas son prácticas y evalúan tanto el producto como el proceso, por tanto, se puede usar  GenAI para experimentar ideas, escribir código inicial, obtener borradores de análisis, etc. Pero, se requiere que los estudiantes critiquen, ajusten y refinen esa producción en su entrega final |
| :---: | :---- | :---- |

**Statement**  
This project aims to **build a sentiment analysis model using supervised learning**.  
You will explore and compare the performance of **Dense Neural Networks**, **vanilla Recurrent Neural Networks (RNN)**, and **LSTM networks** on textual data.  
Finally, you will briefly evaluate the potential of **Transformer-based models** (such as BERT or DistilBERT) for the same task to connect these architectures with the concepts of memory, computation, and generalization introduced through Turing Machines.

**Dataset**  
The dataset is "Sentiment Labelled Sentences Dataset", from the UC Irvine Machine Learning Repository.  
The sentences come from three different websites/fields:

* amazon.com  
* imdb.com  
* yelp.com

Each sentence is labeled as either 1 (for positive) or 0 (for negative).  
For each website, there exist 500 positive and 500 negative sentences.  
This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015\.  
Link to the dataset is: [Sentiment Labelled Sentences Data Set](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)

**Tasks:**

1. **Data Preparation:**  
* Download and load the dataset.  
* Preprocess the text data (tokenization, lowercasing, removing punctuation/stopwords).  
* Split the dataset into training and test sets.  
* Perform an **Exploratory Data Analysis (EDA)** including class balance and sentence length distribution. 

2. **Baseline Model:**  
* Implement a **DummyClassifier** as a baseline.  
* Evaluate it using **accuracy**, **precision**, **recall**, and **F1-score**.

3. **Neural Models:**  
* Implement a **Dense Neural Network**, a **vanilla RNN**, and an **LSTM** model for sentiment classification.  
* Train and tune hyperparameters (e.g., epochs, batch size, hidden units) using tools such as *GridSearchCV* or manual exploration.  
* Evaluate and compare models using **accuracy**, **precision**, **recall**, **F1-score**, and **Cohen’s kappa**.  
4. **Extension (Optional):**  
* Implement a small **Transformer-based model** (e.g., using *Hugging Face’s DistilBERT* or *BERT base uncased*) for comparison.  
* Analyze its performance relative to the recurrent models and discuss how attention mechanisms differ from recurrent or memory-based computation.  
5. **Comparative Analysis:**  
* Compare all models in terms of performance, training complexity, and interpretability.  
* Relate the architectures to **Turing Machine concepts**: memory, sequence processing, and computability.

**Project Deliverables:**

1. **Project Report:** Create a concise report (in English) including the following:  
   1. Introduction and objectives of the project.  
   2. Details of the sentiment identification used and the preprocessing steps.  
   3. Description of the Dense NN, vanilla RNN, and LSTM sentiment classification models, including architecture and training procedure.  
   4. Performance evaluation of the models using standard metrics (i.e., accuracy, precision, recall, F1-score, and kappa).  
   5. Comparative analysis of the models, showcasing any improvements in accuracy.  
   6. Conclusion and future work.  
   7. References.

   The length of the report should be at least two and a maximum of four pages, including figures and references. 

2. **Presentation**: Prepare a short presentation (10 minutes, in English) summarizing your project, emphasizing the problem, methodology, results, and key insights.   
   **Note:** The oral presentation must be technical in nature, focusing on the methodology, models, results, and conclusions presented in the written report. It must not be a product or showcase presentation. The presentation should elaborate on the technical process, explain decisions, discuss results, and demonstrate understanding of the implemented models.   
3. **Project repository.** The code and documents must be placed in a github repository with the following structure:

| `IntegrativeTask2-<team_name>/ │ ├── README.md ├── requirements.txt ├── environment.yml                 # optional, for conda users │ ├── data/ │   ├── raw/                        # original dataset (UCI Sentiment    │                                     Labelled Sentences) │   ├── processed/                  # preprocessed and tokenized data │   └── README.md                   # short description of dataset source and   │                                      preprocessing │ ├── notebooks/ │   ├── 01_eda.ipynb                # Exploratory Data Analysis │   ├── 02_baseline_model.ipynb     # DummyClassifier │   ├── 03_dense_rnn_lstm.ipynb     # Dense NN, RNN, and LSTM models │   ├── 04_transformer_extension.ipynb  # optional Transformer implementation │   └── utils.ipynb                 # helper functions (optional) │ ├── src/ │   ├── __init__.py │   ├── preprocessing.py            # text cleaning, tokenization, embeddings │   ├── models.py                   # model architectures (Dense, RNN, LSTM,  │   │                                 Transformer) │   ├── train.py                    # training routines │   ├── evaluate.py                 # evaluation metrics and comparison │   └── visualize.py                # optional visualizations for results │ ├── outputs/ │   ├── figures/                    # plots, word clouds, charts used in  │                                     report │   ├── metrics/                    # model performance metrics or confusion  │   │                                  matrices │   └── saved_models/               # trained model weights (if small enough) │ ├── docs/ │   ├── report.pdf                  # final written report (English) │   ├── presentation.pdf            # short project presentation │   └── references.bib              # optional bibliography │ ├── prompts/ │   ├── prompt_logs.txt             # AIGen usage in the assignment │   └── AIGen_Interactions.md` `│ └── logs/     ├── training_logs.txt     └── experiment_notes.md` |
| :---- |

**Description of the repository elements:**

| Element | Contents |
| ----- | ----- |
| README.md | Contains project overview, team members, instructions to run the code, and dependencies. |
| requirements.txt / environment.yml | Lists Python libraries needed to run the notebooks and scripts. |
| data/ | Keeps raw and processed data organized. Avoid uploading the raw dataset if it’s too large — include a link instead. |
| notebooks/ | All Jupyter notebooks for EDA, model training, and evaluation. The filenames reflect the workflow order. |
| src/ | Reusable Python modules for preprocessing, model definition, and evaluation. Encourages modular and documented code. |
| outputs/ | Stores generated results such as plots, metrics, and trained model files. |
| docs/ | Includes final deliverables (report and presentation). |
| prompts/ | Contains the report of the interaction with AIG in the assignment |
| logs/ | Optional folder for training or evaluation logs to track experiments. |

**Evaluation Criteria:**  
Your project will be evaluated based on the criteria presented in this [Rubric](https://docs.google.com/spreadsheets/d/1J8fQUD_Klmhh9yazYgH2l_E5Wg1ciW5gZxGLBY0NuAE/edit?usp=sharing):

**Teams**  
This project must involve from 2 to 3 participants (individual projects or projects with more than 3 participants will not be accepted).

You should create your teams using the following GitHub classroom invitation: \[[link](https://classroom.github.com/a/2E33RfU_)\].  Please, fill the following form to register your team \[[link](https://docs.google.com/spreadsheets/d/1TxjfNg9tq3CvmlO1qb-yGL1BPRZqUtmmwjwib2-1BKU/edit?usp=sharing)\].

There should be at least 10 commits with a time difference of 1 hour between each of them. In the repository or project, there must be a directory called 'docs/' in which the report and the short presentation must be placed. Include any necessary clarifications for handling or understanding your project in the readme.md file as additional documentation.

**Deadline: 17th of November, 2025**  