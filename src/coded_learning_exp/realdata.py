from __future__ import annotations

import bz2
import gzip
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .data import SparseRidgeProblem


@dataclass(frozen=True)
class LibsvmDatasetSpec:
    name: str
    url: str
    n_features: int | None = None
    filename: str | None = None


LIBSVM_DATASETS: dict[str, LibsvmDatasetSpec] = {
    "a9a": LibsvmDatasetSpec(
        name="a9a",
        url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
        n_features=123,
    ),
    "w8a": LibsvmDatasetSpec(
        name="w8a",
        url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",
        n_features=300,
    ),
    "rcv1": LibsvmDatasetSpec(
        name="rcv1",
        url="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        n_features=47236,
    ),
}


def make_libsvm_ridge_problem(
    dataset: str,
    n_shards: int,
    l2: float,
    seed: int,
    cache_dir: Path = Path("data") / "libsvm",
    max_samples: int | None = None,
    normalize_rows: bool = True,
    append_bias: bool = True,
    url: str | None = None,
    n_features: int | None = None,
) -> SparseRidgeProblem:
    spec = _resolve_spec(dataset, url=url, n_features=n_features)
    path = download_libsvm_dataset(spec, cache_dir)
    x, y = load_svmlight_file(path, n_features=spec.n_features)
    y = np.where(y > 0, 1.0, -1.0)
    x = x.astype(float).tocsr()
    if normalize_rows:
        x = l2_normalize_rows(x)
    if append_bias:
        bias = sparse.csr_matrix(np.ones((x.shape[0], 1), dtype=float))
        x = sparse.hstack([x, bias], format="csr")
    if max_samples is not None and max_samples < x.shape[0]:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))
        x = x[indices]
        y = y[indices]

    shard_slices = make_even_shard_slices(x.shape[0], n_shards)
    return SparseRidgeProblem(x=x, y=y, shard_slices=shard_slices, l2=l2)


def download_libsvm_dataset(spec: LibsvmDatasetSpec, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = spec.filename or spec.url.rstrip("/").split("/")[-1]
    path = cache_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except Exception:
        # Some LIBSVM mirrors redirect HTTP to HTTPS with certificate chains that
        # are not available in minimal Python installations. The dataset is
        # public and cached locally after download, so fall back to an unverified
        # context rather than requiring machine-specific CA setup.
        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, timeout=120, context=context)

    with response, tmp_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp_path.replace(path)
    return path


def load_svmlight_file(path: Path, n_features: int | None = None) -> tuple[sparse.csr_matrix, np.ndarray]:
    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    labels: list[float] = []
    max_col = -1

    row_id = 0
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            parts = line.split()
            labels.append(float(parts[0]))
            for token in parts[1:]:
                if ":" not in token:
                    continue
                feature, value = token.split(":", 1)
                col = int(feature) - 1
                if col < 0:
                    raise ValueError(f"LIBSVM feature indices must be 1-based: {token}")
                rows.append(row_id)
                cols.append(col)
                data.append(float(value))
                max_col = max(max_col, col)
            row_id += 1

    inferred_features = max_col + 1
    if n_features is None:
        n_features = inferred_features
    elif inferred_features > n_features:
        n_features = inferred_features
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=(len(labels), n_features),
        dtype=float,
    )
    return matrix, np.asarray(labels, dtype=float)


def l2_normalize_rows(x: sparse.csr_matrix) -> sparse.csr_matrix:
    squared = x.multiply(x).sum(axis=1)
    norms = np.sqrt(np.asarray(squared).ravel())
    inv = np.zeros_like(norms)
    nonzero = norms > 0
    inv[nonzero] = 1.0 / norms[nonzero]
    return sparse.diags(inv).dot(x).tocsr()


def make_even_shard_slices(n_samples: int, n_shards: int) -> list[slice]:
    boundaries = np.linspace(0, n_samples, n_shards + 1, dtype=int)
    return [slice(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_shards)]


def _resolve_spec(
    dataset: str,
    url: str | None,
    n_features: int | None,
) -> LibsvmDatasetSpec:
    if url is not None:
        return LibsvmDatasetSpec(name=dataset, url=url, n_features=n_features)
    try:
        return LIBSVM_DATASETS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(LIBSVM_DATASETS))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available built-ins: {available}; "
            "or pass --url."
        ) from exc


def _open_text(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")
