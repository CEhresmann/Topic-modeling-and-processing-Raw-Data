"""
This script performs topic modeling on a corpus of text documents.
"""

import argparse

import gensim
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
import yaml
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(file_path, text_column):
    """Loads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        data = data.dropna(subset=[text_column])
        print("Данные успешно загружены.")
        return data
    except FileNotFoundError:
        print(f"Файл не найден по пути: {file_path}.")
        return None


def download_nltk_data():
    """Downloads the 'punkt' tokenizer for NLTK if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Загрузка 'punkt' для NLTK...")
        nltk.download("punkt")
        print("'punkt' успешно загружен.")


def load_stop_words(stop_words_path):
    """Loads stop words from a file."""
    try:
        with open(stop_words_path, encoding="utf-8") as f:
            return [word.strip() for word in f.readlines()]
    except FileNotFoundError:
        print(f"Файл стоп-слов не найден: {stop_words_path}. Используется пустой список.")
        return []


def preprocess_text(text, filter_words, morph_parser):
    """Preprocesses a single text document."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [word for word in tokens if word not in filter_words]
    lemmatized_tokens = [morph_parser.parse(word)[0].normal_form for word in clean_tokens]
    return lemmatized_tokens


def create_dictionary_and_corpus(processed_texts):
    """Creates a Gensim dictionary and corpus."""
    gensim_dictionary = gensim.corpora.Dictionary(processed_texts)
    gensim_dictionary.filter_extremes(no_above=0.1, no_below=20)
    gensim_dictionary.compactify()
    corpus = [gensim_dictionary.doc2bow(text) for text in processed_texts]
    return gensim_dictionary, corpus


def train_lda_model(corpus, dictionary, num_topics, passes, random_state):
    """Trains an LDA model."""
    return gensim.models.LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=random_state,
    )


def evaluate_coherence(model, texts, dictionary, coherence_measure, title="Coherence Score"):
    """Evaluates the coherence of an LDA model."""
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_measure,
    )
    coherence_score = coherence_model.get_coherence()
    print(f"\n{title}: {coherence_score}")


def find_optimal_topics(
    dictionary,
    corpus,
    texts,
    max_topics,
    start_topics,
    step_topics,
    measure,
    random_state,
):
    """Finds the optimal number of topics by calculating coherence scores."""
    coherence_values = []
    topic_numbers = range(start_topics, max_topics, step_topics)
    for num_topics in topic_numbers:
        model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            passes=10,
            num_topics=num_topics,
            random_state=random_state,
        )
        coherence_model = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence=measure
        )
        coherence_values.append(coherence_model.get_coherence())
        print(f"Расчет для {num_topics} тем завершен.")

    plt.plot(topic_numbers, coherence_values)
    plt.xlabel("Количество тем")
    plt.ylabel(f"Метрика когерентности ({measure})")
    plt.title("Оценка когерентности для разного числа тем")
    plt.show()


def visualize_topics(model, corpus, dictionary, output_path):
    """Visualizes topics using pyLDAvis and saves to HTML."""
    vis_data = gensimvis.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, output_path)
    print(f"Визуализация сохранена в: {output_path}")


def get_dominant_topic(text_processed, lda_model):
    """Determines the dominant topic for a single document."""
    if not text_processed:
        return [None, None]
    bow = lda_model.id2word.doc2bow(text_processed)
    topics = lda_model.get_document_topics(bow)
    if not topics:
        return [None, None]
    dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0]
    return [dominant_topic[0], dominant_topic[1]]


def assign_topics_to_documents(data, text_column, lda_model):
    """Assigns a dominant topic to each document."""
    topic_data = data[text_column].apply(lambda x: get_dominant_topic(x, lda_model))
    data["dominant_topic"] = [item[0] for item in topic_data]
    data["topic_probability"] = [item[1] for item in topic_data]
    return data


def analyze_results(data, category_column=None):
    """Analyzes and visualizes the distribution of topics."""
    if category_column and category_column in data.columns:
        plt.figure(figsize=(15, 8))
        sns.countplot(
            data=data.dropna(subset=["dominant_topic"]),
            x="dominant_topic",
            hue=category_column,
            dodge=True,
        )
        plt.title("Распределение тем по категориям")
        plt.xlabel("Доминирующая тема")
        plt.ylabel("Количество документов")
        plt.xticks(rotation=45)
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        sns.countplot(
            data=data.dropna(subset=["dominant_topic"]),
            x="dominant_topic",
            color="skyblue",
        )
        plt.title("Общее распределение тем по документам")
        plt.xlabel("Доминирующая тема")
        plt.ylabel("Количество документов")
        plt.show()


def run_topic_modeling(config_param):
    """Runs the complete topic modeling pipeline."""
    data_cfg = config_param["data"]
    proc_cfg = config_param["preprocessing"]
    model_cfg = config_param["model"]
    eval_cfg = config_param["evaluation"]
    vis_cfg = config_param["visualization"]

    df = load_data(data_cfg["file_path"], data_cfg["text_column"])
    if df is None:
        return

    download_nltk_data()
    stop_words = load_stop_words(proc_cfg["stop_words_path"])
    punctuation = proc_cfg["punctuation"]
    filter_words = stop_words + list(punctuation)
    morph_parser = MorphAnalyzer()
    df["text_processed"] = df[data_cfg["text_column"]].apply(
        lambda x: preprocess_text(x, filter_words, morph_parser)
    )

    dictionary, corpus = create_dictionary_and_corpus(df["text_processed"])

    lda_model = train_lda_model(
        corpus,
        dictionary,
        model_cfg["num_topics"],
        model_cfg["passes"],
        model_cfg["random_state"],
    )
    lda_model.print_topics()

    evaluate_coherence(
        lda_model,
        df["text_processed"],
        dictionary,
        eval_cfg["coherence_measure"],
    )
    find_optimal_topics(
        dictionary,
        corpus,
        df["text_processed"],
        eval_cfg["max_topics"],
        eval_cfg["start_topics"],
        eval_cfg["step_topics"],
        eval_cfg["coherence_measure"],
        model_cfg["random_state"],
    )

    visualize_topics(lda_model, corpus, dictionary, vis_cfg["output_html_path"])

    df = assign_topics_to_documents(df, "text_processed", lda_model)

    analyze_results(df, data_cfg.get("category_column"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config_main = load_config(args.config)
    run_topic_modeling(config_main)
