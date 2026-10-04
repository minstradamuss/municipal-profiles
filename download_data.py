"""Восстановление точной версии входных файлов по SHA-256.

Первичный источник — СберИндекс. Версия зеркала закреплена commit SHA.
Проверка TLS включена; при корпоративном CA задайте REQUESTS_CA_BUNDLE.
"""
import hashlib
import json
from pathlib import Path
import requests


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'data/sources.json').read_text(encoding='utf-8'))
    dest=root/'data/raw'; dest.mkdir(parents=True,exist_ok=True)
    for item in manifest['files']:
        path=dest/item['file']
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
            print('Проверен:',path.name); continue
        temporary=path.with_suffix(path.suffix+'.part')
        digest=hashlib.sha256()
        try:
            with requests.get(item['url'],stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                with temporary.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):
                        f.write(chunk); digest.update(chunk)
            if digest.hexdigest()!=item['sha256']:
                raise ValueError(f'Источник изменился: {item["file"]}')
            temporary.replace(path)
        finally:
            if temporary.exists(): temporary.unlink()
        print('Скачан:',path.name)


if __name__=='__main__': main()
