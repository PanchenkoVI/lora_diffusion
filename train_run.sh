#!/bin/bash

set -e

rm -rf src/data/my_object_512
rm -rf venv

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "========================================"
echo -e "${GREEN}Настройка проекта LoRA Training${NC}"
echo "========================================"

if ! command -v python3.9 &> /dev/null; then
    echo -e "${RED}Python 3.9 не найден!${NC}"
    exit 1
fi

echo -e "${GREEN}Создаю виртуальное окружение...${NC}"
python3.9 -m venv venv

echo -e "${GREEN}Активирую окружение...${NC}"
source venv/bin/activate

echo -e "${GREEN}Устанавливаю зависимости...${NC}"
pip install --upgrade pip
pip install poetry==1.7.0
poetry install --no-root

echo -e "${GREEN}Проверяю pre-commit (опционально)...${NC}"
poetry run pre-commit run --all-files || true

echo -e "${GREEN}Получаю данные (DVC / fallback)...${NC}"
poetry run python - <<EOF
from src.data.data_manager import DataManager

dm = DataManager()
if not dm.download_data():
    raise RuntimeError("Не удалось получить данные")
EOF

echo -e "${GREEN}Запускаю обучение...${NC}"
echo "========================================"
HYDRA_FULL_ERROR=1 poetry run python train.py
