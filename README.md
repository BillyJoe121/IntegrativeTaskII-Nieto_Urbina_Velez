# IntegrativeTaskII-Nieto_Urbina_Velez

This project aims to build and evaluate several neural network models for sentiment analysis on a labeled dataset.

## 🚀 Setup and Execution Workflow

Follow these steps to set up the environment and run the data processing pipeline.

### 1\. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2\. Create and Activate the Virtual Environment

It is crucial to use a virtual environment to manage project dependencies.

**Create the environment (run once):**

```bash
python -m venv venv
```

**Activate the environment:**

  * **On Windows (PowerShell/CMD):**
    ```bash
    .\venv\Scripts\activate
    ```
  * **On macOS/Linux (Bash/Zsh):**
    ```bash
    source venv/bin/activate
    ```

You will see `(venv)` prefixed to your terminal prompt once it's active.

### 3\. Install Requirements

Install all necessary Python libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4\. Run the EDA Notebook

This step loads the raw data, performs the full Exploratory Data Analysis, and saves the cleaned dataset.

1.  Launch Jupyter Notebook from your terminal (make sure your `venv` is active):
    ```bash
    jupyter notebook
    ```
2.  In the Jupyter interface, navigate to the `notebooks/` directory.
3.  Open `01_eda.ipynb`.
4.  Run all cells from top to bottom.
5.  This will generate a new file: `data/processed/cleaned_data.csv`.

### 5\. Run the Preprocessing Script

This final script takes the cleaned data, applies advanced text preprocessing (lemmatization), splits the data into training and test sets, and saves them.

Run the following command from your **terminal's root directory** (not from inside the `notebooks` or `src` folder):

```bash
python src/preprocessing.py
```

This will use `cleaned_data.csv` to generate two new files:

  * `data/processed/train_data.csv`
  * `data/processed/test_data.csv`

-----

**Project Status:** You are now ready to proceed to the baseline model (`02_baseline_model.ipynb`).
