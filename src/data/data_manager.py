from pathlib import Path
import subprocess
import requests
import zipfile
import tarfile


class DataManager:
    def __init__(self, data_path: str = "src/data/my_object_512"):
        self.data_path = Path(data_path)
        self.dvc_file = self.data_path.with_suffix(".dvc")

    def download_data(self) -> bool:
        self.data_path.mkdir(parents=True, exist_ok=True)

        if self._data_exists():
            print(f"Данные уже есть: {self.data_path}")
            return True

        if self._pull_via_dvc():
            return True

        print("DVC pull не сработал, скачиваю из открытого источника")
        return self._download_from_public_source()

    def _data_exists(self) -> bool:
        return self.data_path.exists() and any(self.data_path.iterdir())

    def _pull_via_dvc(self) -> bool:
        if not self.dvc_file.exists():
            print(".dvc файл не найден, пропускаю dvc pull")
            return False

        print("Пытаюсь скачать данные через DVC...")
        result = subprocess.run(
            ["dvc", "pull", str(self.dvc_file)], capture_output=True, text=True
        )

        if result.returncode == 0:
            print("Данные успешно получены через DVC")
            return True

        print(f"Ошибка dvc pull:\n{result.stderr}")
        return False

    def _download_from_public_source(self) -> bool:
        api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/a2_bgE1W3E_rqA"

        print("Получаю ссылку для скачивания...")
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        download_url = r.json()["href"]
        archive_path = self.data_path / "dataset.zip"

        print("Скачиваю архив...")
        with requests.get(download_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Распаковываю архив...")
        self._extract_archive(archive_path)
        archive_path.unlink(missing_ok=True)

        if not self._data_exists():
            print("Данные не появились после распаковки")
            return False

        print("Данные успешно скачаны из открытого источника")
        return True

    def _extract_archive(self, path: Path):
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                zf.extractall(self.data_path)
        elif path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(self.data_path)
        else:
            raise ValueError(f"Неизвестный формат архива: {path.name}")
