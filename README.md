# Классификация эмоционального окраса текстовых сообщений

Основано на использовании subword-эмбеддингов (bpemb), Tensorflow + Keras.

Более подробная информация о разработке в файле sentiment_analysis.ipynb.

F1 метрика на тестовом наборе данных составляет *0.7128*

### Запуск проекта

docker build --tag classifier:latest

docker run --publish 5000:5000 classifier

### Использование API

**POST localhost:5000/classify**

*Параметры запроса:*
body :: text (string) - текст для классификации.

**Возвращаемые данные - json-объект**

*Параметры возвращаемого объекта:*
result (string) - результат классификации, где возможны варианты "positive", "neutral", "negative"

## Лицензия
MIT
