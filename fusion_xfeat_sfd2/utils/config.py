import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ['.yml', '.yaml']:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path):
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix in ['.yml', '.yaml']:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


def config_hash(cfg: Dict[str, Any]) -> str:
    # 深拷贝并排序
    data = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(data.encode('utf-8')).hexdigest()[:10]


def save_hash_record(cfg: Dict[str, Any], out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = config_hash(cfg)
    with open(out_dir / 'config_hash.txt', 'w', encoding='utf-8') as f:
        f.write(h + '\n')
    return h
