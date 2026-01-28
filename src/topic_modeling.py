# -*- coding: utf-8 -*-
"""
Этот скрипт выполняет тематическое моделирование на основе корпуса текстовых документов.

Пример использования:
    python topicModelingScript.py --config_path /path/to/your/config.yaml

Параметры:
    --config_path (str): Путь к файлу конфигурации .yaml.

Файл конфигурации (config.yaml) должен содержать следующие разделы и параметры:
    data:
      file_path (str): Путь к CSV-файлу с данными.
      text_column (str): Название колонки с текстами.
      category_column (str, optional): Название колонки с категориями для анализа.

    preprocessing:
      stop_words_path (str): Путь к файлу со стоп-словами.
      punctuation (str): Строка со знаками пунктуации для удаления.

    model:
      num_topics (int): Количество тем для LDA-модели.
      passes (int): Количество проходов по корпусу при обучении модели.
      random_state (int): Фиксированное значение для воспроизводимости результатов.

    evaluation:
      coherence_measure (str): Метрика для оценки когерентности (например, 'c_v').
      max_topics (int): Максимальное количество тем для поиска оптимального значения.
      start_topics (int): Начальное количество тем для поиска.
      step_topics (int): Шаг для увеличения количества тем.

    visualization:
      output_html_path (str): Путь для сохранения HTML-файла с визуализацией pyLDAvis.

Этот скрипт выполняет следующие шаги:
1. Загрузка данных из CSV-файла.
2. Предобработка текста: токенизация, удаление стоп-слов и пунктуации, лемматизация.
3. Создание словаря и корпуса для модели Gensim.
4. Обучение модели LDA
5. Оценка модели с использованием метрики когерентности для нахождения оптимального числа тем.
6. Визуализация тем с помощью pyLDAvis.
7. Определение доминирующей темы для каждого документа.
8. Анализ распределения тем, в том числе по категориям, если они указаны.
"""

import argparse
import os

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
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(file_path, text_column):
    try:
        data = pd.read_csv(file_path)
        data = data.dropna(subset=[text_column])
        print("Данные успешно загружены.")
        return data
    except FileNotFoundError:
        print(f"Файл не найден по пути: {file_path}.")
        return None


def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        print("Загрузка 'punkt' для NLTK...")
        nltk.download("punkt")
        print("'punkt' успешно загружен.")


def load_stop_words(stop_words_path):
    try:
        with open(stop_words_path, "r", encoding="utf-8") as f:
            return [word.strip() for word in f.readlines()]
    except FileNotFoundError:
        print(
            f"Файл стоп-слов не найден: {stop_words_path}. Используется пустой список."
        )
        return []


def preprocess_text(text, filter_words, morph_parser):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [word for word in tokens if word not in filter_words]
    lemmatized_tokens = [
        morph_parser.parse(word)[0].normal_form for word in clean_tokens
    ]
    return lemmatized_tokens


def create_dictionary_and_corpus(processed_texts):
    gensim_dictionary = gensim.corpora.Dictionary(processed_texts)
    gensim_dictionary.filter_extremes(no_above=0.1, no_below=20)
    gensim_dictionary.compactify()
    corpus = [gensim_dictionary.doc2bow(text) for text in processed_texts]
    return gensim_dictionary, corpus


def train_lda_model(corpus, dictionary, num_topics, passes, random_state):
    return gensim.models.LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=random_state,
    )


def evaluate_coherence(
    model, texts, dictionary, coherence_measure, title="Coherence Score"
):
    """
    Оценивает когерентность модели LDA.

    :param model: Обученная модель LDA.
    :param texts: Предобработанные тексты.
    :param dictionary: Словарь.
    :param coherence_measure: Метрика когерентности.
    :param title: Заголовок для вывода.
    """
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
    """
    Ищет оптимальное количество тем, вычисляя когерентность для разного числа тем.

    :param dictionary: Словарь.
    :param corpus: Корпус.
    :param texts: Тексты.
    :param max_topics: Максимальное число тем.
    :param start_topics: Начальное число тем.
    :param step_topics: Шаг.
    :param measure: Метрика когерентности.
    :param random_state: Состояние для воспроизводимости.
    """
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
    """
    Визуализирует темы с помощью pyLDAvis и сохраняет в HTML.

    :param model: Обученная модель LDA.
    :param corpus: Корпус.
    :param dictionary: Словарь.
    :param output_path: Путь для сохранения HTML-файла.
    """
    vis_data = gensimvis.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, output_path)
    print(f"Визуализация сохранена в: {output_path}")


def get_dominant_topic(text_processed, lda_model):
    """
    Определяет доминирующую тему для одного документа.

    :param text_processed: Предобработанный текст.
    :param lda_model: Обученная модель LDA.
    :return: Список [номер темы, вероятность].
    """
    if not text_processed:
        return [None, None]
    bow = lda_model.id2word.doc2bow(text_processed)
    topics = lda_model.get_document_topics(bow)
    if not topics:
        return [None, None]
    dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0]
    return [dominant_topic[0], dominant_topic[1]]


def assign_topics_to_documents(data, text_column, lda_model):
    """
    Присваивает каждому документу доминирующую тему.

    :param data: DataFrame.
    :param text_column: Колонка с обработанным текстом.
    :param lda_model: Модель LDA.
    :return: DataFrame с новыми колонками 'dominant_topic' и 'topic_probability'.
    """
    topic_data = data[text_column].apply(lambda x: get_dominant_topic(x, lda_model))
    data["dominant_topic"] = [item[0] for item in topic_data]
    data["topic_probability"] = [item[1] for item in topic_data]
    return data


def analyze_results(data, category_column=None):
    """
    Анализирует и визуализирует распределение тем.

    :param data: DataFrame с результатами.
    :param category_column: Колонка с категориями (опционально).
    """
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


def run_topic_modeling(config):
    data_cfg = config["data"]
    proc_cfg = config["preprocessing"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    vis_cfg = config["visualization"]

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
