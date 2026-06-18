# Clothing Category Recognition API

> CNN классифицирует изображения одежды по 10 категориям за < 30 мс —
> автоматический теггинг товаров при загрузке без ручной разметки.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-teal)]()
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)]()
[![Nginx](https://img.shields.io/badge/Nginx-reverse--proxy-009639)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

E-commerce платформы обрабатывают миллионы загрузок товаров ежедневно.
Ручная категоризация одежды — медленная, дорогая и непоследовательная:
разные операторы кладут один и тот же товар в разные категории.
Этот API классифицирует изображение в момент загрузки — без человека в пайплайне.

---

## Быстрый старт

```bash
git clone https://github.com/your-username/clothing-recognition-api
cd clothing-recognition-api
pip install -r ml_req_deploy.txt

python train.py             # → fashion_mnist.pth
docker-compose up --build   # → http://localhost:80
```

Swagger: `http://localhost:8000/docs`

---

## Demo

```bash
curl -X POST "http://localhost:80/predict" \
  -H "accept: application/json" \
  -F "file=@sneaker.jpg"
```

```json
{
  "class_id": 7,
  "label": "Sneaker",
  "confidence": 0.96
}
```

---

## Результаты

| Модель                          | Accuracy | Улучшение vs baseline |
|---------------------------------|----------|-----------------------|
| Random classifier (10 классов)  | 10%      | baseline              |
| Linear classifier (flatten)     | 85%      | +75%                  |
| **Custom CNN (2-block)**        | **90%**  | **+80%**              |

**Почему кастомный CNN, а не ResNet / EfficientNet:**
- Fashion-MNIST — 28×28 grayscale, не RGB фото реального мира
- Pretrained ResNet обучен на ImageNet (RGB, 224×224) — дообучение избыточно
- 2-block CNN весит 2.1 МБ, инференс < 30 мс на CPU
- 90% на сбалансированном датасете достаточно для production-теггинга

---

## Датасет

- **Источник:** Fashion-MNIST (Zalando Research) — загружается через `torchvision`
- **Объём:** 70 000 grayscale изображений (60K train / 10K test)
- **Размер:** 28×28 px, 1 канал
- **Баланс:** 6 000 примеров на класс — ресэмплинг не нужен

| ID | Класс       | ID | Класс       |
|----|-------------|----|-------------|
| 0  | T-shirt     | 5  | Sandal      |
| 1  | Trouser     | 6  | Shirt       |
| 2  | Pullover    | 7  | Sneaker     |
| 3  | Dress       | 8  | Bag         |
| 4  | Coat        | 9  | Ankle boot  |

---

## Архитектура

    POST /predict  (multipart/form-data, file=image)

│

▼

    Pillow          ← конвертация RGB/RGBA → grayscale, resize 28×28

│

▼

    ToTensor()      ← нормализация [0,255] → [0.0, 1.0]

│

▼

    Conv2d(1→64) + ReLU + MaxPool2d(2)

│

▼

    Flatten + Linear(12544→128) + ReLU + Linear(128→10)

│

▼

    Softmax → argmax    ← class_id + label + confidence

│

▼

    {"class_id": 7, "label": "Sneaker", "confidence": 0.96}


**Ключевые решения:**

Реальные изображения приходят в RGB, RGBA и разных размерах →
`Grayscale() + Resize((28,28))` в inference-трансформе устраняет
ошибки несовпадения размерности без дополнительной обработки на стороне клиента.

`map_location=device` при `torch.load()` — модель обученная на GPU
загружается на CPU-сервере без ошибок. Поведение идентично на обоих окружениях.

---

## Деплой

Два контейнера через Docker Compose:

| Сервис        | Порт | Роль                          |
|---------------|------|-------------------------------|
| `fashion_api` | 8000 | FastAPI + Uvicorn, инференс   |
| `nginx`       | 80   | Reverse proxy                 |

Health check: `GET http://localhost:8000/docs`
(interval: 30s, retries: 5) — Nginx не принимает трафик до готовности API.

```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image>

Response: {"class_id": int, "label": str, "confidence": float}
```

---

## Стек

| Слой       | Технологии                              |
|------------|-----------------------------------------|
| ML         | PyTorch, torchvision, Pillow, NumPy     |
| API        | FastAPI, Uvicorn, Pydantic              |
| Deploy     | Docker, Docker Compose, Nginx           |
| Monitoring | FastAPI `/docs`, Docker health check    |

---

## Что дальше (Roadmap)

- [ ] `confidence` порог — отклонять предсказания ниже 0.6 с `"label": "uncertain"`
- [ ] Батч-эндпоинт `POST /predict/batch` — обработка очереди товаров за один запрос
- [ ] MLflow: версионирование модели и трекинг метрик
- [ ] Переход на RGB-изображения реальных товаров (fine-tuning на EfficientNet-B0)
- [ ] Data drift мониторинг — алерт при смещении входного распределения

---

## Business Impact

| Задача                              | До                          | После                      |
|-------------------------------------|-----------------------------|----------------------------|
| Теггинг одного товара               | 2–5 мин ручной работы       | < 30 мс на запрос          |
| Последовательность категоризации    | Зависит от оператора        | Детерминированная модель   |
| Масштабирование                     | Линейно к числу операторов  | Горизонтальный Docker-скейл|
| Интеграция в пайплайн               | Кастомная разработка        | Один REST-вызов            |

---

[//]: # (## Автор)
[//]: # (**[Имя]** — [LinkedIn]&#40;https://linkedin.com/in/you&#41; | [GitHub]&#40;https://github.com/you&#41;)
