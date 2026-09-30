# Пошаговая настройка Google Colab (Lab 3)

Инструкция для запуска трёх экспериментов на бесплатном GPU.  
**Рекомендуемый путь:** датасет качается в Colab автоматически, на Drive заливаете только код (или используете Git).

Общая документация: [INSTRUCTIONS.md](INSTRUCTIONS.md).

---

## Что понадобится

- Аккаунт Google
- Браузер (Chrome удобнее)
- На домашнем ПК: проект `CV` с лабой (для сборки zip и отчёта после обучения)
- **Не нужно:** заливать на Drive Kaggle `cifar-10/test/` (сотни тысяч файлов без меток — для лабы не используется)

Оценка времени: **~15 минут настройка** + **несколько часов обучение** (3 варианта × до 80 эпох с early stopping).

---

## Часть A. Подготовка на Windows (один раз)

### Шаг A1. Соберите архив с кодом

В PowerShell:

```powershell
cd C:\Users\steps\Projects\CV
.\scripts\pack_for_colab.ps1
```

Появится файл:

`C:\Users\steps\Downloads\CV-lab3-project.zip` (несколько МБ, без данных и `.venv`).

### Шаг A2. (Опционально) Выложите код в GitHub

Если есть репозиторий — в Colab можно обойтись без zip: укажете URL в `GIT_REPO_URL`.  
Иначе переходите к шагу A3.

### Шаг A3. Загрузите zip на Google Drive

1. Откройте [Google Drive](https://drive.google.com).
2. Создайте папку **`CV-lab3`** (в «Мой диск»).
3. Перетащите в неё **`CV-lab3-project.zip`**.

Итог на Drive:

```
Мой диск/CV-lab3/CV-lab3-project.zip
```

Папку **`cifar-10` на Drive класть не нужно**, если будете использовать режим torchvision (шаги B4–B6).

---

## Часть B. Настройка и запуск в Colab

### Шаг B1. Откройте Colab и загрузите ноутбук

1. Перейдите на [colab.research.google.com](https://colab.research.google.com).
2. **Файл → Загрузить блокнот** (Upload notebook).
3. Выберите с диска файл проекта:  
   `C:\Users\steps\Projects\CV\notebooks\colab_lab3.ipynb`

### Шаг B2. Включите GPU

1. Меню **Среда выполнения** (Runtime) → **Сменить тип среды выполнения** (Change runtime type).
2. **Аппаратный ускоритель:** GPU (желательно T4).
3. Нажмите **Сохранить**.

### Шаг B3. Первая ячейка — настройки

Запустите ячейку с комментарием `# --- Настройки ---` (иконка ▶ слева или Shift+Enter).

Проверьте значения:

| Переменная | Рекомендация | Комментарий |
|------------|--------------|-------------|
| `USE_TORCHVISION_DATA` | `True` | Train+test скачаются в Colab (~170 MB) |
| `MOUNT_DRIVE` | `True` | Нужен, если код берёте из zip на Drive |
| `DRIVE_ROOT` | `/content/drive/MyDrive/CV-lab3` | Как на Drive |
| `PROJECT_ZIP` | `{DRIVE_ROOT}/CV-lab3-project.zip` | Путь к zip |
| `GIT_REPO_URL` | `""` | Или URL репозитория вместо zip |
| `EPOCHS_OVERRIDE` | `None` | Полная лаба = 80 эпох из конфига; для проверки поставьте `2` |

Если используете **GitHub**, заполните `GIT_REPO_URL` и ветку `GIT_BRANCH` (`main` / `master`). Тогда zip на Drive не обязателен (можно `MOUNT_DRIVE = False`, если бэкап скачаете только через браузер в конце).

### Шаг B4. Монтирование Google Drive

Запустите следующую ячейку (mount).

- Появится ссылка → разрешите доступ → скопируйте код → вставьте в поле ввода.
- Должно появиться сообщение вроде `Mounted at /content/drive`.

### Шаг B5. Загрузка кода проекта

Запустите ячейку «Project» (распаковка zip или `git clone`).

В выводе должно быть что-то вроде: `Project: /content/CV-lab3`.

**Если ошибка «Нет … CV-lab3-project.zip»** — проверьте шаг A3 и путь `DRIVE_ROOT` / `PROJECT_ZIP`.

### Шаг B6. Проверка данных и GPU

Запустите ячейку с `data_source` и `CUDA`.

При `USE_TORCHVISION_DATA = True` ожидается:

```
Данные: torchvision (train+test скачаются при обучении, без Drive).
data_source = torchvision
CUDA: True NVIDIA ...
```

Если `CUDA: False` — вернитесь к шагу B2 и выберите GPU, затем **Среда выполнения → Перезапустить среду** и пройдите ячейки снова.

### Шаг B7. Установка зависимостей

Запустите ячейку `pip("-e", ".")` — установится пакет `cv-lab3` и зависимости (1–3 минуты).

### Шаг B8. Обучение (три эксперимента)

Запустите большую ячейку с циклом по `aug_variant_2`, `_6`, `_7`.

- При первом запуске скачается CIFAR-10 (прогресс-бар ~170 MB).
- Каждый вариант печатает блоки Architecture / Hyperparameters / эпохи.
- После каждого варианта (если `MOUNT_DRIVE = True`) копия чекпоинтов уходит в  
  `Мой диск/CV-lab3/colab_outputs/`.

**Сессия Colab может оборваться** (лимит бесплатного тарифа). Тогда:

- скачайте уже готовые чекпоинты с Drive из `colab_outputs/`;
- или уменьшите объём: в первой ячейке `EPOCHS_OVERRIDE = 2` для проверки;
- или гоняйте по одному конфигу, перезапуская среду между вариантами (осторожно: ячейка обучения запускает все три подряд — для по одному лучше выполнить на ПК локально одну команду `train.py` в отдельной ячейке вручную).

Одна команда вручную в новой ячейке Colab (пример, только вариант 2):

```python
!cd /content/CV-lab3 && python scripts/train.py --config configs/aug_variant_2.yaml --no-mlflow --download-test --data-source torchvision
```

### Шаг B9. (Опционально) Демо аугментаций

По умолчанию `RUN_DEMO_PREPROCESSING = False`.  
Для демо картинок поставьте `True` в первой ячейке, перезапустите ячейки с настроек и выполните ячейку demo (нужен Kaggle train на Drive или доработка demo под torchvision — на Colab с torchvision demo может не найти PNG; для отчёта демо удобнее сделать на ПК: `demo_preprocessing.py`).

### Шаг B10. Скачать результаты

Запустите последнюю ячейку (zip + `files.download`).

- В браузере начнётся загрузка **`lab3_download.zip`**.
- При `MOUNT_DRIVE = True` копия также в  
  `Мой диск/CV-lab3/colab_outputs/lab3_download_YYYYMMDD_HHMMSS.zip`.

В архиве: `checkpoints/`, при наличии — `artifacts/`.

---

## Часть C. После Colab — на домашнем ПК

### Шаг C1. Распакуйте результаты

Распакуйте `lab3_download.zip` **в корень** проекта `C:\Users\steps\Projects\CV`, чтобы появились:

```
checkpoints/aug_table_2/...
checkpoints/aug_table_6/...
checkpoints/aug_table_7/...
```

### Шаг C2. Соберите отчёт

```powershell
cd C:\Users\steps\Projects\CV
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv run python scripts/generate_report.py
```

Отчёт: `reports/lab3_report.md`.

### Шаг C3. (Опционально) Демо предобработки на ПК

Если на ПК есть Kaggle `cifar-10`:

```powershell
uv run python scripts/demo_preprocessing.py
```

---

## Альтернатива: Kaggle train с Google Drive

Используйте только если нужны именно PNG + `trainLabels.csv` с Kaggle.

1. На Drive: `CV-lab3/cifar-10/trainLabels.csv` и `CV-lab3/cifar-10/train/train/*.png`.
2. **Не копируйте** `cifar-10/test/`.
3. В Colab: `USE_TORCHVISION_DATA = False`.
4. Test по-прежнему подтянется через `--download-test` (torchvision), не с Drive.

---

## Частые проблемы

| Проблема | Решение |
|----------|---------|
| Нет GPU / `CUDA: False` | Runtime → GPU, перезапуск среды |
| Не находит zip на Drive | Проверьте имя папки `CV-lab3` и файл `CV-lab3-project.zip` |
| Обрыв сессии | Брать чекпоинты из `colab_outputs` на Drive; гонять варианты по одному |
| Очень долго | Нормально для 80 эпох × 3; на GPU всё равно быстрее, чем CPU дома |
| Хочу только проверить | `EPOCHS_OVERRIDE = 2` в первой ячейке |

---

## Краткая шпаргалка (чеклист)

- [ ] `pack_for_colab.ps1` → zip на Drive в `CV-lab3/`
- [ ] Загрузить `colab_lab3.ipynb` в Colab
- [ ] Runtime → **GPU**
- [ ] `USE_TORCHVISION_DATA = True`
- [ ] Mount Drive → распаковка проекта → `pip install -e .`
- [ ] Ячейка обучения (3 варианта)
- [ ] Скачать `lab3_download.zip` → распаковать на ПК → `generate_report.py`
