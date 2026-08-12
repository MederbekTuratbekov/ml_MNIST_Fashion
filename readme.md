# Fashion MNIST Clothing Classifier

> CNN классифицирует изображения одежды по 10 категориям за < 30 мс —
> автоматический теггинг товаров при загрузке без ручной разметки.

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal)]()
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)]()
[![Nginx](https://img.shields.io/badge/Nginx-reverse--proxy-009639)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-91.58%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

E-commerce платформы обрабатывают миллионы загрузок товаров ежедневно.
Ручная категоризация одежды — медленная, дорогая и непоследовательная:
разные операторы кладут один и тот же товар в разные категории.
Этот API классифицирует изображение в момент загрузки — без человека в пайплайне.

---

## Структура проекта

    ml_MNIST_Fashion/
    ├── .gitignore
    ├── readme.md
    ├── requirements.txt
    └── MNIST_Fashion/
        ├── Dockerfile
        ├── MNIST_Fashion.ipynb       ← обучение модели (Google Colab, GPU T4)
        ├── docker-compose.yml
        ├── fashion test/             ← тестовые изображения по 10 классам
        │   ├── Ankle boot.png
        │   ├── Bag.png
        │   ├── Coat.png
        │   ├── Dress.png
        │   ├── Pullover.png
        │   ├── Sandal.png
        │   ├── Shirt.png
        │   ├── Sneaker.png
        │   ├── T-shirt_top.png
        │   └── Trouser.png
        ├── main.py                   ← FastAPI сервер + inference
        ├── ml_req_deploy.txt         ← зависимости для Docker
        ├── model_CheckImage_MnistFashion.pth
        └── nginx/
            ├── Dockerfile
            └── nginx.conf

---

## Быстрый старт

```bash
git clone https://github.com/your-username/ml_MNIST_Fashion
cd ml_MNIST_Fashion/MNIST_Fashion

docker-compose up --build   # → http://localhost:80
```

Swagger: `http://localhost:8000/docs`

---

## Demo

```bash
curl -X POST "http://localhost:80/predict" \
  -H "accept: application/json" \
  -F "file=@Sneaker.png"
```

```json
{
  "class_id": 7,
  "class_name": "Sneaker"
}
```

---

## Результаты

| Модель                         | Train Accuracy | Test Accuracy |
|--------------------------------|----------------|---------------|
| Random classifier (10 классов) | 10%            | 10%           |
| Linear (flatten only)          | ~85%           | ~85%          |
| **Custom CNN (1-block)**       | **98.73%**     | **91.58%**    |

Обучение: 20 эпох, Adam lr=0.001, batch_size=32, GPU T4 (Google Colab).

**Почему кастомный CNN, а не ResNet / EfficientNet:**
- Fashion-MNIST — 28×28 grayscale, не RGB фото реального мира
- Pretrained ResNet обучен на ImageNet (RGB, 224×224) — дообучение избыточно
- 1-block CNN весит < 5 МБ, инференс < 30 мс на CPU
- 91.58% на сбалансированном датасете достаточно для production-теггинга

---

## Датасет

- **Источник:** Fashion-MNIST (Zalando Research) — загружается через `torchvision`
- **Объём:** 70 000 grayscale изображений (60K train / 10K test)
- **Размер:** 28×28 px, 1 канал
- **Баланс:** 6 000 примеров на класс — ресэмплинг не нужен

| ID | Класс        | ID | Класс       |
|----|--------------|----|-------------|
| 0  | T-shirt/top  | 5  | Sandal      |
| 1  | Trouser      | 6  | Shirt       |
| 2  | Pullover     | 7  | Sneaker     |
| 3  | Dress        | 8  | Bag         |
| 4  | Coat         | 9  | Ankle boot  |

---

## Архитектура модели

    Conv2d(1→64, kernel=3, padding=1) + BatchNorm2d(64) + ReLU + MaxPool2d(2)
                              ↓
         Flatten → Linear(12544→128) → ReLU → Dropout(0.25) → Linear(128→10)
                              ↓
                    argmax → class_id + class_name

**Ключевые решения:**

`BatchNorm2d` после свёртки — стабилизирует обучение, ускоряет сходимость.

`Dropout(0.25)` в классификаторе — снижает переобучение
(train 98.73% vs test 91.58%, разрыв приемлемый).

`Grayscale() + Resize((28, 28))` в inference-трансформе — реальные изображения
приходят в RGB, RGBA и разных размерах. Конвертация на стороне сервера
устраняет ошибки несовпадения размерности без требований к клиенту.

`map_location=device` при `torch.load()` — модель, обученная на GPU,
загружается на CPU-сервере без ошибок.

---

## Деплой

Два контейнера через Docker Compose:

| Сервис        | Порт | Роль                        |
|---------------|------|-----------------------------|
| `fashion_api` | 8000 | FastAPI + Uvicorn, инференс |
| `nginx`       | 80   | Reverse proxy               |

Health check на `/docs` (interval: 30s, retries: 5) —
Nginx не принимает трафик до готовности API.

```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image>

Response: {"class_id": int, "class_name": str}
```

---

## Стек

| Слой    | Технологии                           |
|---------|--------------------------------------|
| ML      | PyTorch 2.2 (CPU), torchvision, Pillow |
| API     | FastAPI, Uvicorn, Pydantic           |
| Deploy  | Docker, Docker Compose, Nginx        |
| Обучение | Google Colab (GPU T4), Jupyter      |

---

## Business Impact

| Задача                           | До                         | После                      |
|----------------------------------|----------------------------|----------------------------|
| Теггинг одного товара            | 2–5 мин ручной работы      | < 30 мс на запрос          |
| Последовательность категоризации | Зависит от оператора       | Детерминированная модель   |
| Масштабирование                  | Линейно к числу операторов | Горизонтальный Docker-скейл|
| Интеграция в пайплайн            | Кастомная разработка       | Один REST-вызов            |

---

## Что дальше (Roadmap)

- [ ] `confidence` в ответе — вернуть softmax вероятность вместе с классом
- [ ] Батч-эндпоинт `POST /predict/batch` — обработка очереди за один запрос
- [ ] MLflow — трекинг метрик и версионирование модели
- [ ] Переход на RGB фото реальных товаров (fine-tuning EfficientNet-B0)
- [ ] Data drift мониторинг — алерт при смещении входного распределения

---
