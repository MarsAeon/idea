from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import subprocess
import torch


def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def save_checkpoint(state: Dict[str, Any], path: str | Path, is_best: bool = False, max_keep: int = 5):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 附加 git hash
    state = dict(state)
    if 'git_hash' not in state:
        state['git_hash'] = get_git_hash()
    torch.save(state, path)
    if is_best:
        best_path = path.parent / 'model_best.pt'
        torch.save(state, best_path)
    # 滚动清理
    ckpts = sorted(path.parent.glob('epoch_*.pt'))
    if len(ckpts) > max_keep:
        for p in ckpts[:-max_keep]:
            try:
                p.unlink()
            except Exception:
                pass


def load_checkpoint(path: str | Path, map_location: str | torch.device = 'cpu') -> Dict[str, Any]:
    path = Path(path)
    state = torch.load(path, map_location=map_location)
    return state
