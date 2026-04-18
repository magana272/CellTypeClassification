"""Dataset download utilities.

Two download strategies:
- Async parallel range requests for large S3 files (Allen Brain CSVs).
- Synchronous streaming for figshare/GEO URLs (h5ad benchmarks).
"""
from __future__ import annotations

import asyncio
import gzip
import os
import shutil
import ssl

import aiohttp
import certifi
from tqdm.auto import tqdm

CHUNK_SIZE = 1 << 20        # 1 MiB
RANGE_SIZE = 16 << 20       # 16 MiB per range request
MAX_CONCURRENT_RANGES = 16

# ---- Allen Brain CSV datasets (async parallel download) -------------------

ALLEN_BRAIN_DATASETS = {
    "10x": {
        "matrix.csv":   "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_human_m1_10x/matrix.csv",
        "metadata.csv": "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_human_m1_10x/metadata.csv",
    },
    "smartseq": {
        "smartseq_data.csv": "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_human_ctx_smart-seq/matrix.csv",
        "smartseq_meta.csv": "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_human_ctx_smart-seq/metadata.csv",
    },
}

# ---- Benchmark h5ad datasets (sync streaming download) --------------------

H5AD_SOURCES: dict[str, str] = {
    # name -> URL (figshare ndownloader handles redirects on GET)
    "pancreas":      "https://ndownloader.figshare.com/files/24539828",
    "tabula_muris":  "https://ndownloader.figshare.com/files/13092380",  # gzipped
    "lung":          "https://ndownloader.figshare.com/files/24539942",
    # PBMC uses scanpy built-in — no URL needed
}



def download_url(url: str, dest: str) -> None:
    """Stream *url* to *dest* with a progress bar.  Handles redirects."""
    import requests

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        tqdm.write(f"[skip] {os.path.basename(dest)} already exists")
        return

    tqdm.write(f"Downloading {os.path.basename(dest)} ...")
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest + ".part"
    with open(tmp, "wb") as f, tqdm(
        total=total or None, unit="B", unit_scale=True,
        desc=os.path.basename(dest),
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))
    os.replace(tmp, dest)
    tqdm.write(f"[done] {os.path.basename(dest)}")


def download_h5ad(url: str, dest: str) -> None:
    """Download an h5ad file, auto-decompressing if gzipped."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        tqdm.write(f"[skip] {os.path.basename(dest)} already exists")
        return

    # Download to a temp path first
    tmp = dest + ".dl"
    download_url(url, tmp)

    # Check for gzip magic bytes
    with open(tmp, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        tqdm.write("Decompressing gzipped h5ad ...")
        with gzip.open(tmp, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(tmp)
    else:
        os.replace(tmp, dest)



async def _content_length(session: aiohttp.ClientSession, url: str) -> int:
    if not url.startswith(("http://", "https://")):
        return os.path.getsize(url) if os.path.exists(url) else 0
    async with session.head(url, allow_redirects=True) as resp:
        resp.raise_for_status()
        return int(resp.headers.get("Content-Length", 0))


def _pwrite(path: str, data: bytes, offset: int) -> None:
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)


async def _fetch_range(
    session: aiohttp.ClientSession,
    url: str, tmp: str,
    start: int, end: int,
    bar: tqdm, sem: asyncio.Semaphore,
) -> None:
    async with sem:
        headers = {"Range": f"bytes={start}-{end}"}
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.read()
        await asyncio.to_thread(_pwrite, tmp, data, start)
        bar.update(len(data))


async def _download_file(
    session: aiohttp.ClientSession,
    url: str, dest: str, size: int,
    bar: tqdm, sem: asyncio.Semaphore,
) -> None:
    name = os.path.basename(dest)
    if os.path.exists(dest):
        tqdm.write(f"[skip] {name} already exists")
        return
    if not url.startswith(("http://", "https://")):
        if not os.path.exists(url):
            tqdm.write(f"[warn] {name}: source not found at {url}")
            return
        tqdm.write(f"[copy] {name} <- {url}")
        await asyncio.to_thread(_copy_file, url, dest)
        return

    tmp = dest + ".part"
    if size > 0:
        with open(tmp, "wb") as f:
            f.truncate(size)
        ranges = [
            (s, min(s + RANGE_SIZE, size) - 1)
            for s in range(0, size, RANGE_SIZE)
        ]
        await asyncio.gather(
            *(_fetch_range(session, url, tmp, s, e, bar, sem) for s, e in ranges)
        )
    else:
        async with session.get(url) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dest)
    tqdm.write(f"[done] {name}")


def _copy_file(src: str, dest: str) -> None:
    with open(src, "rb") as fi, open(dest, "wb") as fo:
        while True:
            chunk = fi.read(CHUNK_SIZE)
            if not chunk:
                return
            fo.write(chunk)


async def _download_allen_brain_async(root: str = "data") -> None:
    """Download Allen Brain CSV datasets via async range requests."""
    timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(
        ssl=ssl_ctx,
        limit=MAX_CONCURRENT_RANGES * 2,
        limit_per_host=MAX_CONCURRENT_RANGES,
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        pending: list[tuple[str, str]] = []
        for subdir, files in ALLEN_BRAIN_DATASETS.items():
            target_dir = os.path.join(root, subdir)
            os.makedirs(target_dir, exist_ok=True)
            for fname, url in files.items():
                dest = os.path.join(target_dir, fname)
                if not os.path.exists(dest):
                    pending.append((url, dest))

        if not pending:
            tqdm.write("[skip] all Allen Brain files already present")
            return

        sizes = await asyncio.gather(
            *(_content_length(session, url) for url, _ in pending)
        )
        sem = asyncio.Semaphore(MAX_CONCURRENT_RANGES)
        with tqdm(
            total=sum(sizes) or None,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc="downloading",
        ) as bar:
            await asyncio.gather(
                *(
                    _download_file(session, url, dest, size, bar, sem)
                    for (url, dest), size in zip(pending, sizes)
                )
            )



def download_data(root: str = "data") -> None:
    """Download Allen Brain CSV datasets (async parallel)."""
    asyncio.run(_download_allen_brain_async(root))


# Keep old name for backward compat
download_data_async = _download_allen_brain_async

# Alias used by cell_load.py
download_url_to_file = download_url


if __name__ == "__main__":
    download_data()
