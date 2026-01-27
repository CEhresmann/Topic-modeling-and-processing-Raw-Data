"""
Код, из блокнота Тематическое_моделирование.ipynb
"""

# ## 1. Установка и импорт библиотек
# In[ ]:
get_ipython().system(
    "pip install pyldavis pymorphy2 pandas gensim nltk seaborn matplotlib"
)
import os
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
from nltk import download as nltk_download
from pymorphy2 import MorphAnalyzer
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import seaborn as sns


# ## 2. Загрузка данных
# In[ ]:
# Укажите путь к вашему файлу
file_path = "data.csv"
# Укажите имя колонки с текстами
text_column = "text"

try:
    data = pd.read_csv(file_path)
    # Удаляем строки, где текстовая колонка пустая
    data = data.dropna(subset=[text_column])
    print("Данные успешно загружены.")
    data.head()
except FileNotFoundError:
    print(f"Файл не найден по пути: {file_path}. Пожалуйста, укажите правильный путь.")
    # Создадим пример DataFrame для демонстрации
    data = pd.DataFrame(
        {
            text_column: [
                "Тематическое моделирование — это статистический метод для выявления скрытых тем в наборе документов.",
                "Латентное размещение Дирихле (LDA) является популярным алгоритмом для тематического моделирования.",
                "Предобработка текста, такая как лемматизация и удаление стоп-слов, важна для качества модели.",
                "Визуализация тем помогает в их интерпретации и оценке.",
                "Оценка качества модели может производиться с помощью метрики когерентности.",
            ],
            "category": ["ML", "ML", "NLP", "Viz", "ML"],
        }
    )
    print("Создан демонстрационный набор данных.")


# ## 3. Предобработка текста
# In[ ]:
nltk_download("punkt")
try:
    get_ipython().system(
        "wget https://raw.githubusercontent.com/dhhse/dh2020/master/data/stop_ru.txt"
    )
    with open("stop_ru.txt", "r", encoding="utf-8") as stop_ru:
        rus_stops = [word.strip() for word in stop_ru.readlines()]
except Exception as e:
    print(f"Не удалось загрузить стоп-слова: {e}. Используется пустой список.")
    rus_stops = []

punctuation = "!\\"  # $%&'()*+,-./:;<=>?@[\\]^_`{|}~—»«...–"
filter_words = rus_stops + list(punctuation)


# In[ ]:
parser = MorphAnalyzer()


# In[ ]:
def preprocess(input_text, filter_list, morph_parser):
    """
    Функция для предобработки текста.
    :param input_text: Входной текст для очистки и лемматизации.
    :param filter_list: Список стоп-слов и пунктуации для удаления.
    :param morph_parser: Экземпляр MorphAnalyzer.
    :return: Очищенный и лемматизированный текст в виде списка токенов.
    """
    if not isinstance(input_text, str):
        return []
    text = input_text.lower()
    tokenized_text = word_tokenize(text)
    clean_text = [word for word in tokenized_text if word not in filter_list]
    lemmatized_text = [morph_parser.parse(word)[0].normal_form for word in clean_text]
    return lemmatized_text


# In[ ]:
data["text_processed"] = data[text_column].apply(
    lambda x: preprocess(x, filter_words, parser)
)
data.head()


# ## 4. Создание словаря и корпуса
# In[ ]:
gensim_dictionary = gensim.corpora.Dictionary(data["text_processed"])
# no_above: отсекаем слова, которые встречаются в более чем 10% документов
# no_below: отсекаем слова, которые встречаются менее чем в 20 документах
gensim_dictionary.filter_extremes(no_above=0.1, no_below=20)
gensim_dictionary.compactify()
print(gensim_dictionary)


# In[ ]:
corpus = [gensim_dictionary.doc2bow(text) for text in data["text_processed"]]


# ## 5. Построение модели LDA
# In[ ]:
lda_20 = gensim.models.LdaMulticore(
    corpus, num_topics=20, id2word=gensim_dictionary, passes=10, random_state=6457
)


# In[ ]:
lda_20.print_topics()


# ## 6. Оценка модели
# In[ ]:
coherence_model_lda = CoherenceModel(
    model=lda_20,
    texts=data["text_processed"],
    dictionary=gensim_dictionary,
    coherence="c_v",
)
coherence_lda = coherence_model_lda.get_coherence()
print("\nCoherence Score: ", coherence_lda)


# In[ ]:
def coherence_score(
    dictionary, corpus, texts, max_topics, start=2, step=3, measure="c_v"
):
    """
    Вычисляет метрику когерентности для разного числа тем и строит график.
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, max_topics, step):
        model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            passes=10,
            num_topics=num_topics,
            random_state=6457,
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence=measure
        )
        coherence_values.append(coherencemodel.get_coherence())
        print(f"Расчет для {num_topics} тем завершен.")

    x = range(start, max_topics, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Количество тем")
    plt.ylabel(f"Метрика когерентности ({measure})")
    plt.legend(("coherence_values"), loc="best")
    plt.show()

    return model_list, coherence_values


# In[ ]:
# Этот процесс может занять много времени!
model_list_cv, coherence_values_cv = coherence_score(
    dictionary=gensim_dictionary,
    corpus=corpus,
    texts=data["text_processed"],
    start=2,
    max_topics=30,
    step=3,
)


# In[ ]:
optimal_num_topics = 11  # ЗАМЕНИТЕ НА ВАШЕ ОПТИМАЛЬНОЕ ЧИСЛО ТЕМ
final_lda = gensim.models.LdaMulticore(
    corpus,
    num_topics=optimal_num_topics,
    id2word=gensim_dictionary,
    passes=15,  # Можно увеличить для финальной модели
    random_state=6457,
)


# ## 7. Визуализация тем
# In[ ]:
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(final_lda, corpus, gensim_dictionary)


# In[ ]:
vis


# ## 8. Определение доминирующей темы для документов
# In[ ]:
def get_topic(text_processed, lda_model):
    """
    Назначает документу наиболее вероятный топик.
    """
    if not text_processed:
        return [None, None]
    bag_of_words = lda_model.id2word.doc2bow(text_processed)
    topics = lda_model.get_document_topics(bag_of_words)
    if not topics:
        return [None, None]
    dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0]
    return [dominant_topic[0], dominant_topic[1]]


# In[ ]:
topic_data = data["text_processed"].apply(lambda x: get_topic(x, final_lda))
data["dominant_topic"] = [item[0] for item in topic_data]
data["topic_probability"] = [item[1] for item in topic_data]
data.head()


# ## 9. Анализ результатов
# In[ ]:
# Пример анализа. Замените 'category' на имя вашей категориальной колонки.
category_column = "category"  # ЗАМЕНИТЕ, ЕСЛИ НУЖНО

if category_column in data.columns:
    plt.figure(figsize=(15, 8))
    sns.countplot(data=data, x="dominant_topic", hue=category_column, dodge=True)
    plt.title("Распределение тем по категориям")
    plt.xlabel("Доминирующая тема")
    plt.ylabel("Количество документов")
    plt.xticks(rotation=45)
    plt.show()
else:
    print(f'Колонка "{category_column}" не найдена. Построим общее распределение тем.')
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=data.dropna(subset=["dominant_topic"]), x="dominant_topic", color="skyblue"
    )
    plt.title("Общее распределение тем по документам")
    plt.xlabel("Доминирующая тема")
    plt.ylabel("Количество документов")
    plt.show()
