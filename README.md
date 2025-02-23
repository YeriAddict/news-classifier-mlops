# Fake News Classifier

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li>
      <a href="#running-in-development-mode">Running in Development Mode</a>
      <ul>
        <li><a href="#using-python">Using Python</a></li>
        <li><a href="#using-docker">Using Docker</a></li>
      </ul>
    </li>
    <li>
      <a href="#pipeline-description">Pipeline Description</a>
      <ul>
        <li>
          <a href="#data-ingestion">Data Ingestion</a>
          <ul>
            <li><a href="#current-implementation">Current Implementation</a></li>
            <li><a href="#scaling-and-improvement-suggestions">Improvements</a></li>
          </ul>
        </li>
        <li>
          <a href="#exploratory-data-analysis">Exploratory Data Analysis</a>
          <ul>
            <li><a href="#current-implementation">Current Implementation</a></li>
            <li><a href="#scaling-and-improvement-suggestions">Improvements</a></li>
          </ul>
        </li>
        <li>
          <a href="#data-preprocessing">Data Preprocessing</a>
          <ul>
            <li><a href="#current-implementation">Current Implementation</a></li>
            <li><a href="#scaling-and-improvement-suggestions">Improvements</a></li>
          </ul>
        </li>
        <li>
          <a href="#features-engineering">Features Engineering</a>
          <ul>
            <li><a href="#current-implementation">Current Implementation</a></li>
            <li><a href="#scaling-and-improvement-suggestions">Improvements</a></li>
          </ul>
        </li>
        <li>
          <a href="#model-development">Model Development</a>
          <ul>
            <li><a href="#current-implementation">Current Implementation</a></li>
            <li><a href="#scaling-and-improvement-suggestions">Improvements</a></li>
          </ul>
        </li>
        <li>
          <a href="#model-deployment-and-integration">Model Deployment and Integration</a>
          <ul>
            <li><a href="#current-implementation">Current Implementation</a></li>
            <li><a href="#scaling-and-improvement-suggestions">Improvements</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

---

## Overview

Media plays a vital role in today's information landscape, providing a primary source for news. For reputable agencies, verifying sources is essential to ensure that the information is reliable and trustworthy. However, manually fact-checking every statement made by public figures can be extremely labor-intensive. This project aims to build and train a machine learning model capable of predicting whether a statement is true or false, effectively functioning as a fake news classifier.

Its main goal is also to focus on the entire pipeline of a machine learning project, from ingesting and processing the data to deploying the model.

---

### Dataset

The model is trained on the **LIAR** dataset, which you can find on [Papers With Code](https://paperswithcode.com/dataset/liar). This dataset contains over 12,800 statements made by politicians, public figures, and others.

#### Dataset Structure

The dataset includes the following columns:

| Column                  | Description                                                                          |
|-------------------------|--------------------------------------------------------------------------------------|
| **id**                  | Unique identifier for each statement.                                                |
| **label**               | The truthfulness label of the statement.                                             |
| **statement**           | The text of the statement.                                                           |
| **subjects**            | Subject(s) related to the statement.                                                 |
| **speaker**             | The individual who made the statement.                                               |
| **speaker_job_title**   | The job title of the speaker.                                                        |
| **state_info**          | Information about the state.                                                         |
| **party_affiliation**   | The political party affiliation of the speaker.                                      |
| **barely_true_counts**  | Count of "barely true" ratings (including the current statement).                    |
| **false_counts**        | Count of "false" ratings.                                                            |
| **half_true_counts**    | Count of "half true" ratings.                                                        |
| **mostly_true_counts**  | Count of "mostly true" ratings.                                                      |
| **pants_on_fire_counts**| Count of "pants on fire" ratings.                                                    |
| **context**             | The context or location where the statement was made.                                |

### Model

The baseline model for this project is a simple **Random Forest Classifier**. With the current hyperparameters (as shown in the snippet below), the model achieves an accuracy of approximately **70%**. Notably, this performance surpasses all the accuracies presented in Table 2 of [this research paper](https://aclanthology.org/W18-5513.pdf), which served as the inspiration for this project.

---

## Running in Development Mode

### Customizing the Config File

The `random_forest.json` file located in the `config` directory is used to define the model architecture, data paths, and hyperparameters for training. You can fine-tune the machine learning model by modifying these parameters.

For example, the current configuration looks like this:

```json
{
    "model": "random_forest",
    "raw_data_paths": {
        "train": "datasets/raw/train.tsv",
        "test": "datasets/raw/test.tsv",
        "val": "datasets/raw/valid.tsv",
        "combined": "datasets/raw/combined.csv"
    },
    "clean_data_paths": {
        "data": {
            "train": "datasets/clean/data/train_data.csv",
            "test": "datasets/clean/data/test_data.csv",
            "val": "datasets/clean/data/val_data.csv"
        },
        "labels": {
            "train": "datasets/clean/labels/train_labels.csv",
            "test": "datasets/clean/labels/test_labels.csv",
            "val": "datasets/clean/labels/val_labels.csv"
        }
    },
    "featurizer_path": "cache/featurizer.pkl",
    "model_path": "cache/random_forest_model.pkl",
    "params": {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced"
    }
}
```

### Setting up the API

#### Using Python

To deploy the API with Python, just run the following commands from the root directory:

- **Installing dependencies:**

  ```bash
  python -m pip install -r requirements.txt
  ```
  
- **Training the model:**

  ```bash
  python -m news_classifier.run --config-file config/random_forest.json
  ```

- **Hosting the API:**

  ```bash
  python -m uvicorn news_classifier.model_integration.api:app --reload
  ```

#### Using Docker

To deploy the API with Docker, make sure you start Docker and then run the following commands from the root directory:

- **Building the image and training the model**

  ```bash
  docker build -t news-classifier-api .
  ```
  
- **Hosting the API:**

  ```bash
  docker run -p 8000:8000 news-classifier-api
  ```

### Testing the endpoint:

Currently, there is only a POST request called "predict" that takes a text statement as argument and returns a label that is either 0 (False) or 1 (True). To try it, it is possible to access the swagger page dedicated to the API:

```bash
  http://localhost:8000/docs
```

Additionally, you can also run this cURL command:

```bash
  curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"text": "some example string"}'
```

---

## Pipeline Description

### Data Ingestion

#### Current Implementation

Data ingestion is currently based on static, well-prepared TSV files. Initially, the data is split into three files: `train.tsv`, `test.tsv`, and `val.tsv`. However, due to uncertainties about class distribution and other potential issues, these three datasets are merged into a single CSV file. This approach ensures that all available data is considered and allows for a more unified analysis.

#### Scaling and Improvement Suggestions

While the current solution relies on static data files, a more scalable approach would be to integrate live, up-to-date data. For example, you could fetch news data from an API such as [NewsAPI](https://newsapi.org/), and then store the data in a data warehouse or cloud storage solution. This setup would allow the model pipeline to be continuously updated with fresh data, thereby improving its relevance and accuracy over time.

---

### Exploratory Data Analysis

#### Current Implementation

Exploratory Data Analysis (EDA) is a crucial step that informs subsequent decisions in the pipeline. It involves examining key aspects of the data, such as class distribution, data quality, and potential anomalies. This analysis helps determine whether to perform a standard or stratified split and informs the appropriate cleaning methods.

For a detailed examination of the data, please refer to the `exploratory_data_analysis.ipynb` notebook in the `notebooks` directory.

#### Scaling and Improvement Suggestions

- Automate the EDA process to generate updated reports as new data is ingested.
- Incorporate interactive visualization tools (e.g., Plotly or Dash) for a more dynamic exploration of data trends.
- Integrate statistical tests to identify and address potential data anomalies proactively.

---

### Data Preprocessing

#### Current Implementation

This phase focuses on cleaning the dataset and removing any potential noise that could negatively impact the model's performance. The process includes:

- Canonicalizing and cleaning several columns.
- Addressing a slight imbalance in class distribution identified during the EDA phase by applying stratified sampling.
- Separating and saving the training, testing, and validation datasets along with their corresponding labels.

#### Scaling and Improvement Suggestions

For a scalable solution, one of the main challenges is labeling each news article fetched from the API. A potential approach is to implement pseudo-labeling:

- **Pseudo-labeling:** Train a robust initial model to generate labels for newly fetched data automatically. Although this method might introduce some noise, it can serve as an effective solution for automating the labeling process, provided the model’s performance is satisfactory.

This approach can enable the pipeline to continuously ingest and process live data while maintaining an acceptable level of accuracy.

---

### Features Engineering

#### Current Implementation

The current features engineering approach transforms raw claim data into a comprehensive feature set that can be effectively used for machine learning. The process integrates both structured and unstructured data by employing the following techniques:

- **Manual Feature Extraction:**  
  Key categorical and numerical attributes (such as party affiliation, speaker, and state information) are extracted. Numerical credibility metrics (e.g., counts like "false_counts") are discretized into bins, reducing noise and variability in the data. This transformation is similar to the process facilitated by a dictionary vectorizer.

- **Textual Feature Extraction:**  
  The textual content of the claims is processed using a TF-IDF vectorizer. This technique converts text into numerical features that reflect the importance of words and phrases across the dataset, capturing the semantic information contained in the statements.

- **Integrated Pipeline:**  
  Both manual and textual features are combined using a unified pipeline, which leverages tools such as scikit-learn's `FeatureUnion` and `Pipeline`. This integration ensures that all features are aligned and prepared in a consistent format, with additional steps (like imputation) applied to handle any missing values.

Overall, this dual approach—combining a `DictVectorizer` for structured data with a `TfidfVectorizer` for text—creates a robust and informative feature representation that supports effective model training.

#### Scaling and Improvement Suggestions

- **Advanced Textual Representations:**  
  Integrate more sophisticated NLP techniques such as word embeddings (e.g., Word2Vec or GloVe) or contextual embeddings (e.g., BERT or RoBERTa) to capture deeper semantic nuances in the text.

- **Enhanced Feature Selection and Dimensionality Reduction:**  
  Utilize methods like Principal Component Analysis (PCA) or recursive feature elimination to reduce the dimensionality of the feature space and eliminate redundant features, thereby improving model performance and training efficiency.

- **Hyperparameter Tuning for Vectorizers:**  
  Experiment with different configurations for the `TfidfVectorizer` and `DictVectorizer`—for example, adjusting n-gram ranges, minimum document frequency, or exploring alternative weighting schemes—to identify the optimal setup for your dataset.

- **Integration of External Data Sources:**  
  Enrich the feature set by incorporating external metadata or supplementary information, such as sentiment scores or fact-checking signals, which can provide additional context and enhance predictive accuracy.

---

### Model Development

#### Current Implementation

The current implementation centers on a Random Forest model integrated with a custom feature engineering pipeline. It works by loading a pre-trained featurizer, transforming raw claim data into numerical features, and then training a Random Forest classifier using parameters specified in a configuration file. The model evaluation is performed using standard metrics such as accuracy, F1 score, ROC AUC, and a confusion matrix to assess performance.

#### Scaling and Improvement Suggestions

- **Model Benchmarking:**  
  Evaluate alternative models (e.g., logistic regression, SVM, or neural networks) to identify the best performing architecture for the task.

- **Hyperparameter Tuning:**  
  Use techniques such as k-fold cross-validation to fine-tune hyperparameters and optimize model performance.

- **Advanced Ensemble Techniques:**  
  Consider combining multiple models through ensemble methods to potentially improve overall prediction accuracy.

---

### Model Deployment and Integration

#### Current Implementation

The current deployment is local, with the entire pipeline containerized into a Docker image that hosts the API built using FastAPI. Continuous Integration and Continuous Deployment (CI/CD) pipelines, implemented via GitHub Actions, automatically run tests to ensure that the API functions as expected.

#### Scaling and Improvement Suggestions for Model Deployment

Since the pipeline is already containerized, deploying the API to cloud services is straightforward. This would allow external users to call the endpoint for veracity predictions on demand. Future improvements could include implementing advanced orchestration tools (e.g., Kubernetes) for better scalability and reliability.

---

### Model Monitoring

While model monitoring was not a primary focus of this project, it remains an essential component in production. Monitoring enables continuous tracking of model performance, hyperparameters, and data drift in a centralized manner. Currently, the project leverages MLflow for basic monitoring, but more robust solutions (e.g., Prometheus with Grafana or commercial offerings like Datadog) could be integrated to provide enhanced insights and alerts.

---

## References

- Research Paper: [Fake News Detection using Stance Classification](https://aclanthology.org/W18-5513.pdf)
