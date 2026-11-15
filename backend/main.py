"""Planet X & SPHEREx - FastAPI proxy for the IRSA SPHEREx archive.

Serves the embedded frontend and renders FITS cutouts to PNG in memory.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
import numpy as np
from astropy.io import fits
from astropy.io.votable import parse as parse_votable
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV?"
COLLECTIONS = ("spherex_dr1", "spherex_qr3")
ALLOWED_HOST = "irsa.ipac.caltech.edu"
ALLOWED_PATH = "/ibe/data/spherex/"
USER_AGENT = "planet-x-spherex/1.0 (NASA Space Apps Challenge 2026)"
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

# SPHEREx detector wavelength edges in microns (DR1/QR3 level-2 products).
BANDS = {
    1: (0.7455, 1.1167),
    2: (1.1033, 1.6499),
    3: (1.6301, 2.4346),
    4: (2.4027, 3.8473),
    5: (3.8113, 4.4300),
    6: (4.4115, 5.0096),
}
COLORMAPS = ("gray", "inferno", "viridis")
MAX_FRAMES = 24
MAX_GIF_FRAMES = 24
RENDER_PX = 400

GAIA_TAP = "https://gaia.ari.uni-heidelberg.de/tap/sync"
GAIA_RADIUS_DEG = 0.008
APERTURE_PX = 1.8
AB_ZERO_JY = 3631.0


class TTLCache:
    """Tiny in-memory LRU cache with a time-to-live."""

    def __init__(self, ttl: float, maxsize: int) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.monotonic() + self.ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)


QUERY_CACHE = TTLCache(ttl=1800, maxsize=256)
PNG_CACHE = TTLCache(ttl=86_400, maxsize=192)
RESOLVE_CACHE = TTLCache(ttl=86_400, maxsize=256)
PHOT_CACHE = TTLCache(ttl=86_400, maxsize=512)
SCIENCE_CACHE = TTLCache(ttl=3600, maxsize=256)

# Last good result per coordinate, used when IRSA fails.
STALE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
STALE_TTL = 6 * 3600
STALE_MAX = 256


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(35.0, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    )
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="Planet X & SPHEREx",
    description="IRSA SIA v2 proxy and FITS cutout renderer for SPHEREx DR1/QR3.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- helpers
def _cell(row: Any, name: str) -> Any:
    try:
        value = row[name]
    except (KeyError, IndexError, TypeError):
        return None
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iso_time(value: Any) -> tuple[str | None, float | None]:
    """Return (ISO-8601 UTC string, MJD) from a SIA t_min value."""
    if value is None:
        return None, None
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
            mjd = (dt - MJD_EPOCH).total_seconds() / 86400.0
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), mjd
        except ValueError:
            return text, None
    mjd = _as_float(value)
    if mjd is None:
        return None, None
    if 20_000 < mjd < 100_000:
        dt = MJD_EPOCH + timedelta(days=mjd)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), mjd
    if mjd > 1_000_000_000:  # Unix seconds, just in case
        dt = datetime.fromtimestamp(mjd, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), (dt - MJD_EPOCH).total_seconds() / 86400.0
    return text, mjd


def _band_number(bandpass: Any, em_min: float | None, em_max: float | None) -> int | None:
    match = re.search(r"D([1-6])", str(bandpass or ""), re.IGNORECASE)
    if match:
        return int(match.group(1))
    if em_min and em_max:
        center = (em_min + em_max) / 2.0
        for band, (lo, hi) in BANDS.items():
            if lo * 1e-6 <= center <= hi * 1e-6:
                return band
    return None


async def _get(client: httpx.AsyncClient, url: str, params: Any = None, timeout: float = 35.0) -> httpx.Response:
    last_error: Exception | None = None
    retry_status = (429, 500, 502, 503, 504)
    attempts = 4
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params, timeout=timeout)
            if response.status_code in retry_status and attempt < attempts - 1:
                await asyncio.sleep(0.6 * (2**attempt) + random.uniform(0, 0.4))
                continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                await asyncio.sleep(0.6 * (2**attempt) + random.uniform(0, 0.4))
    if isinstance(last_error, httpx.HTTPStatusError):
        status = last_error.response.status_code
        raise HTTPException(502, f"IRSA returned HTTP {status}. Please retry in a moment.") from last_error
    raise HTTPException(504, "IRSA is taking too long to answer. Please retry with a smaller radius.") from last_error


async def _resolve_name(client: httpx.AsyncClient, name: str) -> dict[str, Any]:
    """Resolve an object name to ICRS coordinates with CDS Sesame (SIMBAD/NED)."""
    key = name.casefold()
    cached = RESOLVE_CACHE.get(key)
    if cached is not None:
        return cached

    url = SESAME_URL + quote(name, safe="")
    response: httpx.Response | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await client.get(url, timeout=15.0)
            if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                await asyncio.sleep(0.6)
                continue
            response.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(0.6)

    if response is None or response.status_code >= 400:
        if isinstance(last_error, httpx.TimeoutException):
            raise HTTPException(504, "The name resolver (Sesame) is taking too long. Try again.")
        raise HTTPException(502, "Could not reach the name resolver (Sesame). Try again in a moment.")

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise HTTPException(502, "The name resolver returned an invalid response.") from exc

    candidates: list[dict[str, Any]] = []
    for node in root.iter("Resolver"):
        ra = _as_float(node.findtext("jradeg"))
        dec = _as_float(node.findtext("jdedeg"))
        if ra is None or dec is None or not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            continue
        resolver = (node.get("name") or "Sesame").split(",")[0].strip()
        if resolver.startswith("S="):
            resolver = resolver[2:].strip()
        candidates.append(
            {
                "name": (node.findtext("oname") or name).strip(),
                "ra": ra,
                "dec": dec,
                "resolver": resolver or "Sesame",
                "otype": (node.findtext("otype") or "").strip() or None,
            }
        )

    if not candidates:
        raise HTTPException(404, f'No object was found for "{name}". Try another name or use coordinates.')

    # Prefer SIMBAD, then NED, then any other Sesame resolver.
    candidates.sort(
        key=lambda item: 0 if "Simbad" in item["resolver"] else 1 if "NED" in item["resolver"] else 2
    )
    target = candidates[0]
    payload = {"query": name, **target}
    RESOLVE_CACHE.set(key, payload)
    return payload


async def _query_collection(
    client: httpx.AsyncClient, collection: str, ra: float, dec: float, radius: float
) -> list[dict[str, Any]]:
    params = [
        ("COLLECTION", collection),
        ("POS", f"CIRCLE {ra:.6f} {dec:.6f} {radius:.6f}"),
        ("DPTYPE", "image"),
        ("CALIB", "2"),
        ("RESPONSEFORMAT", "VOTABLE"),
        ("MAXREC", "600"),
    ]
    response = await _get(client, SIA_URL, params=params, timeout=40.0)
    content = response.content
    if not content.lstrip().startswith((b"<?xml", b"<VOTABLE")):
        raise ValueError(f"{collection}: unexpected response from IRSA")
    try:
        table = parse_votable(io.BytesIO(content)).get_first_table().to_table(use_names_over_ids=True)
    except Exception as exc:  # noqa: BLE001 - VOTable parser errors vary
        raise ValueError(f"{collection}: could not parse VOTable") from exc

    frames: list[dict[str, Any]] = []
    for row in table:
        access_url = _cell(row, "access_url")
        if not access_url:
            continue
        em_min = _as_float(_cell(row, "em_min"))
        em_max = _as_float(_cell(row, "em_max"))
        band = _band_number(_cell(row, "energy_bandpassname"), em_min, em_max)
        date_iso, mjd = _iso_time(_cell(row, "t_min"))
        wavelength = (em_min + em_max) / 2.0 * 1e6 if em_min and em_max else None
        obs_id = str(_cell(row, "obs_id") or "")
        frames.append(
            {
                "id": f"{collection}:{obs_id}:D{band}" if band else f"{collection}:{obs_id}",
                "obs_id": obs_id,
                "collection": str(_cell(row, "obs_collection") or collection),
                "date": date_iso,
                "mjd": round(mjd, 6) if mjd is not None else None,
                "band": band,
                "wavelength_um": round(wavelength, 3) if wavelength else None,
                "em_min_um": round(em_min * 1e6, 4) if em_min else None,
                "em_max_um": round(em_max * 1e6, 4) if em_max else None,
                "access_url": str(access_url),
                "ra": _as_float(_cell(row, "s_ra")),
                "dec": _as_float(_cell(row, "s_dec")),
                "fov_deg": _as_float(_cell(row, "s_fov")),
                "target_name": _cell(row, "target_name"),
                "calib_level": _cell(row, "calib_level"),
                "access_estsize": _as_float(_cell(row, "access_estsize")),
            }
        )
    return frames


async def _observations(client: httpx.AsyncClient, ra: float, dec: float, radius: float) -> dict[str, Any]:
    key = f"obs:{ra:.5f}:{dec:.5f}:{radius:.5f}"
    cached = QUERY_CACHE.get(key)
    if cached is not None:
        return cached

    results = await asyncio.gather(
        *(_query_collection(client, collection, ra, dec, radius) for collection in COLLECTIONS),
        return_exceptions=True,
    )
    frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    for collection, result in zip(COLLECTIONS, results):
        if isinstance(result, Exception):
            warnings.append(f"{collection}: {result}")
        else:
            frames.extend(result)

    if not frames and len(warnings) == len(COLLECTIONS):
        stale = STALE_CACHE.get(key)
        if stale and time.monotonic() - stale[0] < STALE_TTL:
            cached = dict(stale[1])
            cached["warnings"] = list(cached.get("warnings") or []) + [
                "IRSA is unavailable right now; showing the last successful result for these coordinates."
            ]
            cached["stale"] = True
            return cached
        raise HTTPException(502, "IRSA is unavailable for both SPHEREx collections right now.")

    deduped: dict[str, dict[str, Any]] = {}
    for frame in frames:
        deduped.setdefault(frame["access_url"], frame)
    frames = sorted(deduped.values(), key=lambda f: (f["mjd"] or 0.0, f["band"] or 0))

    bands = []
    for band, (lo, hi) in BANDS.items():
        count = sum(1 for frame in frames if frame["band"] == band)
        if count:
            bands.append({"band": band, "min_um": lo, "max_um": hi, "count": count})

    payload = {
        "success": True,
        "query": {"ra": ra, "dec": dec, "radius": radius, "collections": list(COLLECTIONS)},
        "count": len(frames),
        "bands": bands,
        "frames": frames,
        "warnings": warnings,
        "message": None
        if frames
        else "No calibrated SPHEREx DR1/QR3 images intersect this position. Try a larger radius or another coordinate.",
    }
    QUERY_CACHE.set(key, payload)
    STALE_CACHE[key] = (time.monotonic(), payload)
    while len(STALE_CACHE) > STALE_MAX:
        STALE_CACHE.pop(next(iter(STALE_CACHE)))
    return payload


def _validate_fits_url(fits_url: str) -> str:
    parts = urlsplit(fits_url)
    if parts.scheme != "https" or parts.hostname != ALLOWED_HOST or not parts.path.startswith(ALLOWED_PATH):
        raise HTTPException(400, "Only SPHEREx FITS access URLs from irsa.ipac.caltech.edu are allowed.")
    return fits_url


def _cutout_url(fits_url: str, ra: float, dec: float, size: float) -> str:
    parts = urlsplit(_validate_fits_url(fits_url))
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({"center": f"{ra:.6f},{dec:.6f}deg", "size": f"{size:.6f}deg", "gzip": "false"})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _parse_header(buf: bytes, start: int) -> tuple[dict[str, Any], int] | None:
    """Parse one FITS header block sequence starting at ``start``."""
    pos = start
    cards: dict[str, Any] = {}
    while True:
        if pos + 2880 > len(buf):
            return None
        block = buf[pos : pos + 2880]
        for i in range(0, 2880, 80):
            card = block[i : i + 80]
            if card[:8].strip() == b"END":
                return cards, pos + 2880
            key = card[:8].decode("ascii", "replace").strip()
            if not key or card[8:10] != b"= ":
                continue
            raw = card[10:].split(b"/")[0].strip()
            if raw[:1] == b"'":
                cards[key] = raw.strip(b"' ").decode("ascii", "replace").strip()
                continue
            try:
                cards[key] = int(raw)
            except ValueError:
                try:
                    cards[key] = float(raw.replace(b"D", b"E"))
                except ValueError:
                    cards[key] = raw.decode("ascii", "replace").strip()
        pos += 2880


def _first_image_hdu(buf: bytes) -> tuple[int, int] | None:
    """Locate the first complete 2-D HDU in a partially downloaded FITS file."""
    pos = 0
    while True:
        parsed = _parse_header(buf, pos)
        if parsed is None:
            return None
        cards, data_start = parsed
        naxis = int(cards.get("NAXIS", 0) or 0)
        bitpix = abs(int(cards.get("BITPIX", 8) or 8))
        count = 1
        for axis in range(1, naxis + 1):
            count *= max(1, int(cards.get(f"NAXIS{axis}", 1) or 1))
        data_bytes = bitpix // 8 * count if naxis else 0
        end = data_start + ((data_bytes + 2879) // 2880) * 2880
        if end > len(buf):
            return None
        if naxis >= 2:
            return pos, end
        pos = end


async def _download_cutout(client: httpx.AsyncClient, fits_url: str, ra: float, dec: float, size: float) -> bytes:
    """Stream a cutout and stop once the IMAGE HDU is complete.

    IRSA cuts every extension, so most of the payload is the EPSF table; the
    IMAGE HDU arrives first and the connection is closed there.
    """
    url = _cutout_url(fits_url, ra, dec, size)
    last_status: int | None = None
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            async with client.stream("GET", url, timeout=httpx.Timeout(60.0, connect=10.0)) as response:
                if response.status_code != 200:
                    last_status = response.status_code
                    if response.status_code not in (429, 500, 502, 503, 504):
                        break
                else:
                    buffer = bytearray()
                    async for chunk in response.aiter_bytes(16384):
                        buffer.extend(chunk)
                        if buffer[:2] == b"\x1f\x8b":
                            continue
                        found = _first_image_hdu(buffer)
                        if found is not None:
                            start, end = found
                            if start == 0:
                                return bytes(buffer[:end])
                            return bytes(buffer[:2880]) + bytes(buffer[start:end])

                    data = bytes(buffer)
                    if data[:2] == b"\x1f\x8b":
                        try:
                            data = gzip.decompress(data)
                        except OSError as exc:
                            raise HTTPException(502, "IRSA returned a corrupt compressed cutout.") from exc
                    if data.startswith(b"SIMPLE"):
                        return data
                    last_error = HTTPException(502, "IRSA did not return a FITS cutout for this frame.")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
        if attempt < 2:
            await asyncio.sleep(0.8 * (attempt + 1))

    if last_status is not None and last_status not in (429, 500, 502, 503, 504):
        raise HTTPException(502, f"IRSA cutout request failed (HTTP {last_status}).")
    if last_status in (429, 500, 502, 503, 504):
        raise HTTPException(502, f"IRSA cutout service is busy (HTTP {last_status}). Please retry in a moment.")
    if isinstance(last_error, HTTPException):
        raise last_error
    raise HTTPException(504, "IRSA is taking too long to return this cutout. Please retry.") from last_error


def _read_image_hdu(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with fits.open(io.BytesIO(data), memmap=False, ignore_missing_end=True) as hdul:
            candidates = [hdu for hdu in hdul if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2]
            image_hdu = next((hdu for hdu in candidates if hdu.name.upper() == "IMAGE"), None)
            hdu = image_hdu or (candidates[0] if candidates else None)
            if hdu is None:
                raise ValueError("no 2-D image extension")
            array = np.asarray(hdu.data, dtype=np.float64)
            header = dict(hdu.header)
    except (OSError, ValueError) as exc:
        raise HTTPException(422, "The SPHEREx cutout has no usable 2-D IMAGE extension.") from exc

    array = np.squeeze(array)
    if array.ndim != 2 or array.size < 16:
        raise HTTPException(422, "The SPHEREx cutout is too small to render.")
    return np.flipud(array), header


def _load_cutout_array(data: bytes) -> np.ndarray:
    array, _ = _read_image_hdu(data)

    finite = np.isfinite(array)
    if finite.sum() < 16:
        raise HTTPException(422, "The SPHEREx cutout is empty (all pixels are NaN).")
    values = array[finite]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = mad * 1.4826
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = float(values.std())
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    # Asinh stretch in robust sigmas: dark sky, unclipped bright cores.
    top = max((float(np.percentile(values, 99.9)) - median) / sigma, 8.0)
    top = min(top, 400.0)
    soft = 4.0
    normalized = np.arcsinh(np.maximum(array - median, 0.0) / (sigma * soft)) / math.asinh(top / soft)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32)


@lru_cache(maxsize=8)
def _colormap_lut(name: str) -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255.0).round().astype(np.uint8)


def _square(image: Image.Image, size: int = RENDER_PX) -> Image.Image:
    """Scale the cutout up to a fixed square canvas (IRSA cutouts are tiny)."""
    scale = size / max(image.width, image.height)
    target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(target, Image.Resampling.LANCZOS)
    canvas = Image.new(image.mode, (size, size), 0)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def _render_array(normalized: np.ndarray, colormap: str) -> Image.Image:
    gray = Image.fromarray((normalized * 255.0).astype(np.uint8), "L")
    if colormap == "gray":
        return _square(gray)
    return _square(Image.fromarray(_colormap_lut(colormap)[np.asarray(gray)], "RGB"))


def _png_response(png: bytes) -> Response:
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _png_from_normalized(normalized: np.ndarray, colormap: str) -> bytes:
    image = _render_array(normalized, colormap)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _aperture_photometry(array: np.ndarray, size_deg: float, bunit: str = "") -> dict[str, float] | None:
    if "MJY" not in bunit.upper():
        return None
    height, width = array.shape
    if height < 5 or width < 5:
        return None

    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.hypot(yy - (height - 1) / 2.0, xx - (width - 1) / 2.0)
    aperture = radius <= APERTURE_PX
    outer = min(max(float(radius.max()) - 1.0, APERTURE_PX + 1.0), 8.0)
    inner = max(APERTURE_PX + 1.0, outer * 0.55)
    annulus = (radius >= inner) & (radius <= outer)
    if aperture.sum() < 4 or annulus.sum() < 12:
        return None

    values = np.where(np.isfinite(array), array, np.nan)
    background = float(np.nanmedian(values[annulus]))
    spread = 1.4826 * float(np.nanmedian(np.abs(values[annulus] - background)))
    if not math.isfinite(spread) or spread <= 0:
        spread = float(np.nanstd(values[annulus]))
    if not math.isfinite(spread) or spread <= 0:
        return None

    pixel_deg = size_deg / max(width, height)
    pixel_sr = math.radians(pixel_deg) ** 2
    counts = float(np.nansum(values[aperture] - background))
    flux_jy = counts * pixel_sr * 1e6
    noise = math.sqrt(aperture.sum() * (1.0 + aperture.sum() / annulus.sum()))
    error_jy = spread * pixel_sr * 1e6 * noise

    magnitude = -2.5 * math.log10(flux_jy / AB_ZERO_JY) if flux_jy > 0 else None
    magnitude_error = 1.0857 * error_jy / flux_jy if flux_jy > 0 and error_jy > 0 else None
    return {
        "flux_jy": flux_jy,
        "flux_error_jy": error_jy,
        "mag": magnitude,
        "mag_error": magnitude_error,
        "background": background,
        "pixel_scale_arcsec": pixel_deg * 3600.0,
        "aperture_px": APERTURE_PX,
    }


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    cos_dec = math.cos(math.radians((dec1 + dec2) / 2.0))
    return math.hypot((ra1 - ra2) * cos_dec, dec1 - dec2) * 3600.0


async def _gaia_source(client: httpx.AsyncClient, ra: float, dec: float) -> dict[str, Any]:
    query = (
        "SELECT TOP 20 source_id, ra, dec, parallax, parallax_error, "
        "phot_g_mean_mag, bp_rp, ruwe FROM gaiadr3.gaia_source "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {GAIA_RADIUS_DEG})) = 1"
    )
    payload: dict[str, Any] | None = None
    for attempt in range(2):
        try:
            response = await client.get(
                GAIA_TAP,
                params={"request": "doQuery", "lang": "ADQL", "format": "json", "query": query},
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
            response.raise_for_status()
            payload = response.json()
            break
        except (httpx.HTTPError, ValueError):
            if attempt == 0:
                await asyncio.sleep(1.2)

    if payload is None:
        return {"catalog": "Gaia DR3", "available": False, "reason": "unavailable"}

    columns = [item.get("name") for item in payload.get("metadata", []) if item.get("name")]
    rows = [row if isinstance(row, dict) else dict(zip(columns, row)) for row in payload.get("data", [])]
    best: tuple[float, dict[str, Any]] | None = None
    for row in rows:
        if row.get("ra") is None or row.get("dec") is None:
            continue
        separation = _separation_arcsec(ra, dec, float(row["ra"]), float(row["dec"]))
        if best is None or separation < best[0]:
            best = (separation, row)

    if best is None:
        return {
            "catalog": "Gaia DR3",
            "available": False,
            "reason": "no_source",
            "radius_arcsec": round(GAIA_RADIUS_DEG * 3600.0, 1),
        }

    separation, row = best
    parallax = float(row["parallax"]) if row.get("parallax") is not None else None
    error = float(row["parallax_error"]) if row.get("parallax_error") is not None else None
    ruwe = float(row["ruwe"]) if row.get("ruwe") is not None else None
    snr = parallax / error if parallax and error else None

    distance_pc = 1000.0 / parallax if parallax and parallax > 0 else None
    low = high = None
    if distance_pc is not None and error and parallax > error:
        low = 1000.0 / (parallax + error)
        high = 1000.0 / (parallax - error)

    return {
        "catalog": "Gaia DR3",
        "available": True,
        "source_id": str(row.get("source_id", "")),
        "separation_arcsec": round(separation, 2),
        "parallax_mas": parallax,
        "parallax_error_mas": error,
        "snr": round(snr, 1) if snr else None,
        "distance_pc": distance_pc,
        "distance_ly": distance_pc * 3.26156 if distance_pc else None,
        "distance_pc_low": low,
        "distance_pc_high": high,
        "g_mag": float(row["phot_g_mean_mag"]) if row.get("phot_g_mean_mag") is not None else None,
        "bp_rp": float(row["bp_rp"]) if row.get("bp_rp") is not None else None,
        "ruwe": ruwe,
        "reliable": bool(snr and snr >= 5 and (ruwe is None or ruwe < 1.4)),
    }



# -------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
async def index() -> Response:
    return Response(content=INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "planet-x-spherex",
        "collections": list(COLLECTIONS),
        "sia": SIA_URL,
        "resolver": SESAME_URL,
        "bands": BANDS,
    }


@app.get("/api/resolve")
async def resolve(
    request: Request,
    name: str = Query(..., min_length=1, max_length=80, description="Object name (SIMBAD/NED)"),
) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", name).strip()
    if not cleaned:
        raise HTTPException(400, "Enter an object name.")
    return await _resolve_name(request.app.state.http, cleaned)


@app.get("/api/observations")
async def observations(
    request: Request,
    ra: float = Query(..., ge=0.0, lt=360.0, description="Right ascension in decimal degrees"),
    dec: float = Query(..., ge=-90.0, le=90.0, description="Declination in decimal degrees"),
    radius: float = Query(0.01, gt=0.0, le=0.5, description="Search radius in degrees"),
    band: int | None = Query(None, ge=1, le=6, description="Optional SPHEREx detector band"),
) -> dict[str, Any]:
    payload = await _observations(request.app.state.http, ra, dec, radius)
    frames = payload["frames"]
    if band is not None:
        frames = [frame for frame in frames if frame["band"] == band]

    total = len(frames)
    truncated = total > MAX_FRAMES
    if truncated:
        indices = sorted({round(i) for i in np.linspace(0, total - 1, MAX_FRAMES)})
        frames = [frames[i] for i in indices]

    return {**payload, "count": len(frames), "total": total, "truncated": truncated, "frames": frames}


@app.get("/api/science")
async def science(
    request: Request,
    ra: float = Query(..., ge=0.0, lt=360.0),
    dec: float = Query(..., ge=-90.0, le=90.0),
) -> dict[str, Any]:
    key = f"sci:{ra:.4f}:{dec:.4f}"
    cached = SCIENCE_CACHE.get(key)
    if cached is not None:
        return cached
    payload = await _gaia_source(request.app.state.http, ra, dec)
    SCIENCE_CACHE.set(key, payload)
    return payload


class PhotoFrame(BaseModel):
    fits_url: str = Field(..., min_length=8)
    date: str | None = None
    band: int | None = Field(None, ge=1, le=6)


class PhotometryRequest(BaseModel):
    frames: list[PhotoFrame] = Field(..., min_length=1, max_length=MAX_FRAMES)
    ra: float = Field(..., ge=0.0, lt=360.0)
    dec: float = Field(..., ge=-90.0, le=90.0)
    size: float = Field(0.05, gt=0.0, le=2.0)


@app.post("/api/photometry")
async def photometry(request: Request, body: PhotometryRequest) -> dict[str, Any]:
    client = request.app.state.http
    semaphore = asyncio.Semaphore(4)

    async def point(item: PhotoFrame) -> dict[str, Any]:
        base: dict[str, Any] = {"date": item.date, "band": item.band}
        key = f"phot:{item.fits_url}:{body.ra:.5f}:{body.dec:.5f}:{body.size:.5f}"
        cached = PHOT_CACHE.get(key)
        if cached is not None:
            return {**base, **cached}
        try:
            async with semaphore:
                data = await _download_cutout(client, item.fits_url, body.ra, body.dec, body.size)
            array, header = _read_image_hdu(data)
            measurements = _aperture_photometry(array, body.size, str(header.get("BUNIT", "")))
        except HTTPException as exc:
            return {**base, "error": exc.detail}
        if measurements is None:
            return {**base, "error": "No usable aperture in this frame."}
        PHOT_CACHE.set(key, measurements)
        return {**base, **measurements}

    points = await asyncio.gather(*(point(item) for item in body.frames))
    scales = [p["pixel_scale_arcsec"] for p in points if "pixel_scale_arcsec" in p]
    return {
        "unit": "AB mag (aperture)",
        "aperture_px": APERTURE_PX,
        "pixel_scale_arcsec": round(float(np.median(scales)), 2) if scales else None,
        "points": points,
    }


@app.get("/api/cutout")
async def cutout(
    request: Request,
    fits_url: str = Query(..., min_length=8),
    ra: float = Query(..., ge=0.0, lt=360.0),
    dec: float = Query(..., ge=-90.0, le=90.0),
    size: float = Query(0.05, gt=0.0, le=2.0),
    colormap: str = Query("gray"),
) -> Response:
    fits_url = _validate_fits_url(fits_url)
    colormap = colormap.lower()
    if colormap not in COLORMAPS:
        raise HTTPException(400, f"colormap must be one of {', '.join(COLORMAPS)}")

    key = f"cut:{fits_url}:{ra:.5f}:{dec:.5f}:{size:.5f}:{colormap}"
    cached = PNG_CACHE.get(key)
    if cached is not None:
        return _png_response(cached)

    data = await _download_cutout(request.app.state.http, fits_url, ra, dec, size)
    normalized = _load_cutout_array(data)
    png = _png_from_normalized(normalized, colormap)
    PNG_CACHE.set(key, png)
    return _png_response(png)


@app.get("/api/diff")
async def difference(
    request: Request,
    fits_url_1: str = Query(...),
    fits_url_2: str = Query(...),
    ra: float = Query(..., ge=0.0, lt=360.0),
    dec: float = Query(..., ge=-90.0, le=90.0),
    size: float = Query(0.05, gt=0.0, le=2.0),
    colormap: str = Query("gray"),
) -> Response:
    """Difference of two cutouts. White = brighter in the second frame."""
    if colormap.lower() not in COLORMAPS:
        raise HTTPException(400, f"colormap must be one of {', '.join(COLORMAPS)}")
    client = request.app.state.http
    first, second = await asyncio.gather(
        _download_cutout(client, fits_url_1, ra, dec, size),
        _download_cutout(client, fits_url_2, ra, dec, size),
    )
    a = _render_array(_load_cutout_array(first), "gray").convert("L")
    b = _render_array(_load_cutout_array(second), "gray").convert("L")
    delta = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    scale = float(np.percentile(np.abs(delta), 99.5)) or 1.0
    scaled = np.clip(delta / scale, -1.0, 1.0)
    gray = ((scaled + 1.0) * 127.5).round().astype(np.uint8)
    image = Image.fromarray(gray, "L")
    if colormap.lower() != "gray":
        image = Image.fromarray(_colormap_lut(colormap.lower())[gray], "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


class GifRequest(BaseModel):
    fits_urls: list[str] = Field(..., min_length=2, max_length=MAX_GIF_FRAMES)
    ra: float = Field(..., ge=0.0, lt=360.0)
    dec: float = Field(..., ge=-90.0, le=90.0)
    size: float = Field(0.05, gt=0.0, le=2.0)
    colormap: str = "gray"
    delay_ms: int = Field(400, ge=100, le=5000)


@app.post("/api/gif")
async def export_gif(request: Request, body: GifRequest) -> Response:
    if body.colormap.lower() not in COLORMAPS:
        raise HTTPException(400, f"colormap must be one of {', '.join(COLORMAPS)}")
    client = request.app.state.http
    semaphore = asyncio.Semaphore(4)

    async def frame(url: str) -> Image.Image:
        async with semaphore:
            data = await _download_cutout(client, url, body.ra, body.dec, body.size)
        return _render_array(_load_cutout_array(data), body.colormap.lower()).convert("RGB")

    frames = await asyncio.gather(*(frame(url) for url in body.fits_urls))
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=body.delay_ms,
        loop=0,
        optimize=True,
    )
    return Response(
        content=buffer.getvalue(),
        media_type="image/gif",
        headers={"Content-Disposition": 'attachment; filename="spherex-timelapse.gif"', "Cache-Control": "no-store"},
    )


# Frontend served at "/".
INDEX_HTML = r'''<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="dark" />
    <title>Planet X &amp; SPHEREx — Time-lapse infrarrojo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {
        theme: {
          extend: {
            colors: {
              ink: "#0B0F19",
              panel: "#101725",
              line: "#21304a",
              accent: "#5eead4",
              amber: "#fbbf24",
            },
            fontFamily: {
              sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
              mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
            },
          },
        },
      };
    </script>
    <style>
      :root { color-scheme: dark; }
      html, body { background: #0b0f19; }
      body {
        background-image:
          radial-gradient(1100px 600px at 80% -10%, rgba(45, 212, 191, 0.09), transparent 60%),
          radial-gradient(900px 500px at 10% 0%, rgba(251, 191, 36, 0.06), transparent 55%);
        background-repeat: no-repeat;
      }
      .panel {
        background: rgba(16, 23, 37, 0.86);
        border: 1px solid rgba(33, 48, 74, 0.9);
        box-shadow: 0 18px 50px -30px rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(10px);
      }
      .hud {
        position: absolute;
        border-radius: 0.65rem;
        border: 1px solid rgba(94, 234, 212, 0.22);
        background: rgba(5, 9, 16, 0.62);
        padding: 0.45rem 0.6rem;
        color: #cbd5e1;
        font-size: 10px;
        line-height: 1.35;
        letter-spacing: 0.03em;
        backdrop-filter: blur(7px);
        text-shadow: 0 1px 3px #000;
        pointer-events: none;
      }
      .hud .k { color: #64748b; text-transform: uppercase; font-size: 8px; letter-spacing: 0.14em; }
      .hud .v { color: #e2e8f0; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; }
      .hud .v.cyan { color: #5eead4; }
      .hud .v.amber { color: #fbbf24; }
      .crosshair-line { position: absolute; background: rgba(94, 234, 212, 0.45); box-shadow: 0 0 6px rgba(94, 234, 212, 0.35); }
      #viewerWrap.zoomed { touch-action: none; cursor: grab; }
      #viewerWrap.zoomed.dragging { cursor: grabbing; }
      .zoom-btn {
        display: inline-grid; place-items: center;
        width: 1.7rem; height: 1.7rem;
        border-radius: 0.5rem; border: 1px solid rgba(33, 48, 74, 0.9);
        background: rgba(11, 15, 25, 0.6); color: #cbd5e1;
        font-size: 15px; line-height: 1;
      }
      .zoom-btn:hover { border-color: rgba(94, 234, 212, 0.55); color: #5eead4; }
      .zoom-btn:disabled { opacity: 0.4; cursor: not-allowed; }
      .spinner {
        width: 34px; height: 34px; border-radius: 50%;
        border: 2px solid rgba(94, 234, 212, 0.18);
        border-top-color: #5eead4;
        animation: spin 0.8s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      input[type="range"] { accent-color: #5eead4; }
      input[type="range"]::-webkit-slider-runnable-track { height: 4px; border-radius: 999px; background: #21304a; }
      input[type="range"]::-webkit-slider-thumb { margin-top: -6px; }
      .preset:hover { border-color: rgba(94, 234, 212, 0.55); background: rgba(94, 234, 212, 0.06); }
      .preset.active { border-color: rgba(94, 234, 212, 0.7); background: rgba(94, 234, 212, 0.1); }
      .btn-primary {
        background: linear-gradient(180deg, #5eead4, #2dd4bf);
        color: #052e2b;
        font-weight: 700;
        box-shadow: 0 10px 30px -14px rgba(45, 212, 191, 0.85);
      }
      .btn-primary:hover { filter: brightness(1.06); }
      .btn-primary:disabled { opacity: 0.45; filter: grayscale(0.4); cursor: not-allowed; }
      .blink { background: linear-gradient(180deg, #fbbf24, #f59e0b); color: #251400; }
      .blink:hover { filter: brightness(1.06); }
      .blink.active { background: linear-gradient(180deg, #f87171, #ef4444); color: #fff; }
      .chip { border: 1px solid #21304a; background: rgba(16, 23, 37, 0.7); }
      .fade-in { animation: fade 0.35s ease-out; }
      @keyframes fade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
      .mode-btn {
        border-radius: 0.375rem; padding: 0.25rem 0.6rem;
        font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        color: #94a3b8; transition: color 0.15s, background 0.15s;
      }
      .mode-btn:hover { color: #e2e8f0; }
      .mode-btn.active { background: #182238; color: #5eead4; }
      body[data-mode="casual"] .sci-only { display: none !important; }
      select, input[type="number"], input[type="text"] { color-scheme: dark; }
      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-thumb { background: #21304a; border-radius: 999px; }
    </style>
  </head>
  <body class="min-h-screen font-sans text-slate-200 antialiased" data-mode="casual">
    <header class="sticky top-0 z-40 border-b border-[#21304a]/80 bg-[#0B0F19]/90 backdrop-blur">
      <div class="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-4 py-3">
        <div class="grid h-11 w-11 place-items-center rounded-full bg-[radial-gradient(circle_at_35%_30%,#99f6e4,#0f766e_65%,#022c22)] text-lg text-[#04201d] shadow-[0_0_30px_rgba(45,212,191,0.35)]">◉</div>
        <div class="min-w-0">
          <h1 class="truncate text-lg font-semibold tracking-tight text-white">
            Planet X <span class="text-teal-300">&amp;</span> SPHEREx
          </h1>
          <p class="truncate text-[11px] text-slate-400" data-i18n="app.subtitle">Time-lapse de recortes infrarrojos reales · NASA/IPAC IRSA</p>
        </div>
        <div class="ml-auto flex items-center gap-2">
          <div id="modeSwitch" class="flex gap-0.5 rounded-lg border border-[#21304a] bg-[#101725] p-1" role="group" aria-label="Viewer mode">
            <button type="button" data-mode="casual" class="mode-btn active" aria-pressed="true" data-i18n="mode.casual">Casual</button>
            <button type="button" data-mode="science" class="mode-btn" aria-pressed="false" data-i18n="mode.science">Científico</button>
          </div>
          <a href="/docs" target="_blank" rel="noopener" class="chip rounded-lg px-2.5 py-1.5 text-[11px] font-medium text-slate-300 hover:border-teal-400/50 hover:text-teal-300" data-i18n="nav.api">API docs</a>
          <div id="lang" class="flex gap-0.5 rounded-lg border border-[#21304a] bg-[#101725] p-1" role="group" aria-label="Language">
            <button type="button" data-lang="es" class="rounded-md px-2.5 py-1 text-[11px] font-bold tracking-wide text-slate-400 transition hover:text-slate-100">ES</button>
            <button type="button" data-lang="en" class="rounded-md px-2.5 py-1 text-[11px] font-bold tracking-wide text-slate-400 transition hover:text-slate-100">EN</button>
          </div>
        </div>
      </div>
    </header>

    <main class="mx-auto grid max-w-7xl grid-cols-1 gap-5 px-4 py-5 lg:grid-cols-[360px_minmax(0,1fr)]">
      <aside class="space-y-4">
        <section class="panel fade-in rounded-2xl p-4">
          <h2 class="text-sm font-semibold text-white" data-i18n="search.title">Buscar en el cielo</h2>
          <p class="mt-1 text-[11px] leading-relaxed text-slate-400" data-i18n="search.hint">Coordenadas ICRS/J2000 en grados decimales. Se consultan las colecciones calibradas spherex_dr1 y spherex_qr3.</p>

          <form id="searchForm" class="mt-4 space-y-3" novalidate>
            <label class="block">
              <span class="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="search.name">Nombre del objeto</span>
              <div class="flex gap-2">
                <input id="objectName" type="text" autocomplete="off" spellcheck="false" maxlength="80" placeholder="M42, Betelgeuse, M63…" data-i18n-placeholder="search.namePlaceholder" class="min-w-0 flex-1 rounded-lg border border-[#21304a] bg-[#0B0F19] px-3 py-2 text-sm text-white outline-none placeholder:text-slate-600 focus:border-teal-400/70" />
                <button id="resolveBtn" type="button" class="chip shrink-0 rounded-lg px-3 py-2 text-xs text-slate-200 hover:border-teal-400/50" data-i18n="search.resolve">Resolver</button>
              </div>
            </label>

            <div class="grid grid-cols-2 gap-3">
              <label class="block">
                <span class="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="search.ra">RA (°)</span>
                <input id="ra" type="number" step="0.0001" min="0" max="359.9999" value="269.452" class="w-full rounded-lg border border-[#21304a] bg-[#0B0F19] px-3 py-2 font-mono text-sm text-white outline-none focus:border-teal-400/70" />
              </label>
              <label class="block">
                <span class="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="search.dec">Dec (°)</span>
                <input id="dec" type="number" step="0.0001" min="-90" max="90" value="4.693" class="w-full rounded-lg border border-[#21304a] bg-[#0B0F19] px-3 py-2 font-mono text-sm text-white outline-none focus:border-teal-400/70" />
              </label>
            </div>

            <div class="sci-only grid grid-cols-2 gap-3">
              <label class="block">
                <span class="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="search.fov">FOV</span>
                <select id="fov" class="w-full rounded-lg border border-[#21304a] bg-[#0B0F19] px-3 py-2 text-sm text-white outline-none focus:border-teal-400/70">
                  <option value="0.02">0.02° (1.2′)</option>
                  <option value="0.05" selected>0.05° (3′)</option>
                  <option value="0.1">0.10° (6′)</option>
                </select>
              </label>
              <label class="block">
                <span class="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="search.band">Banda</span>
                <select id="band" class="w-full rounded-lg border border-[#21304a] bg-[#0B0F19] px-3 py-2 text-sm text-white outline-none focus:border-teal-400/70">
                  <option value="auto" data-i18n="band.auto">Auto</option>
                  <option value="all" data-i18n="band.all">Todas</option>
                </select>
              </label>
            </div>

            <label class="block">
              <span class="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="search.colormap">Mapa de color</span>
              <select id="cmap" class="w-full rounded-lg border border-[#21304a] bg-[#0B0F19] px-3 py-2 text-sm text-white outline-none focus:border-teal-400/70">
                <option value="gray" data-i18n="cmap.gray">Escala de grises</option>
                <option value="inferno" data-i18n="cmap.inferno">Inferno</option>
                <option value="viridis" data-i18n="cmap.viridis">Viridis</option>
              </select>
            </label>

            <button id="searchBtn" type="submit" class="btn-primary w-full rounded-xl px-4 py-2.5 text-sm transition disabled:cursor-wait">
              <span data-i18n="search.submit">Buscar observaciones</span>
            </button>
          </form>

          <div id="status" class="mt-3 rounded-lg border border-[#21304a] bg-[#0B0F19]/60 px-3 py-2 text-[11px] leading-relaxed text-slate-400">
            <span data-i18n="status.idle">Elige un objetivo o introduce coordenadas.</span>
          </div>
          <div id="error" class="mt-2 hidden rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-[11px] leading-relaxed text-red-300"></div>
        </section>

        <section class="panel fade-in rounded-2xl p-4">
          <div class="flex items-baseline justify-between gap-2">
            <h2 class="text-sm font-semibold text-white" data-i18n="presets.title">Objetivos de prueba</h2>
            <span class="text-[10px] text-slate-500" data-i18n="presets.hint">Un clic busca</span>
          </div>
          <div id="presets" class="mt-3 grid gap-2"></div>
        </section>

        <section class="panel fade-in rounded-2xl p-4">
          <h2 class="text-sm font-semibold text-white" data-i18n="tips.title">Cómo leer el visor</h2>
          <ul class="mt-2 space-y-2 text-[11px] leading-relaxed text-slate-400">
            <li class="flex gap-2"><span class="mt-0.5 text-teal-300">▸</span><span data-i18n="tips.one">Usa Parpadeo rápido para alternar la primera y la última fecha: un punto que salta es un candidato a objeto móvil.</span></li>
            <li class="flex gap-2"><span class="mt-0.5 text-teal-300">▸</span><span data-i18n="tips.two">Activa Resta para restar el primer fotograma: las estrellas fijas quedan en gris y el movimiento aparece como dipolo blanco/negro.</span></li>
            <li class="sci-only flex gap-2"><span class="mt-0.5 text-teal-300">▸</span><span data-i18n="tips.three">Filtra por banda D1–D6 para evitar parpadeos de brillo entre filtros infrarrojos.</span></li>
            <li class="flex gap-2"><span class="mt-0.5 text-amber-300">▸</span><span data-i18n="tips.four">Esto es un explorador, no una detección validada: verifica siempre en Aladin/VizieR.</span></li>
          </ul>
        </section>
      </aside>

      <section class="min-w-0 space-y-4">
        <div id="viewerWrap" class="panel relative aspect-square w-full overflow-hidden rounded-2xl bg-black">
          <canvas id="view" width="800" height="800" class="block h-full w-full" aria-label="SPHEREx time-lapse viewer"></canvas>

          <div class="hud left-3 top-3 max-w-[54%]">
            <div class="k" data-i18n="hud.mission">Misión</div>
            <div id="hudMission" class="v cyan">SPHEREx —</div>
            <div id="hudDate" class="v mt-1">—</div>
          </div>
          <div class="hud sci-only right-3 top-3 text-right">
            <div class="k" data-i18n="hud.wavelength">Longitud de onda</div>
            <div id="hudWave" class="v amber">— µm</div>
            <div class="k mt-1" data-i18n="hud.fov">Campo</div>
            <div id="hudFov" class="v">—</div>
          </div>
          <div class="hud sci-only bottom-3 left-3">
            <div class="k" data-i18n="hud.coords">Coordenadas</div>
            <div id="hudCoords" class="v">—</div>
            <div id="hudFrame" class="mt-1 text-[10px] text-slate-400">—</div>
          </div>
          <div id="diffBadge" class="hud bottom-3 right-3 hidden border-red-400/40 text-red-300">
            <span class="k text-red-300/70" data-i18n="hud.diffLabel">Modo</span>
            <div class="v" style="color:#fca5a5" data-i18n="hud.diff">RESTA</div>
            <div id="diffCount" class="mt-1 text-[10px] leading-tight text-amber-300/90"></div>
          </div>

          <div id="crosshair" class="pointer-events-none absolute inset-0">
            <div id="crosshairV" class="crosshair-line left-1/2 top-0 h-full w-px"></div>
            <div id="crosshairH" class="crosshair-line left-0 top-1/2 h-px w-full"></div>
            <div id="crosshairDot" class="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border border-teal-300/40"></div>
          </div>

          <div id="loader" class="absolute inset-0 grid place-items-center bg-[#05090f]/82 p-6 backdrop-blur-sm">
            <div class="w-full max-w-sm text-center">
              <div class="mx-auto spinner"></div>
              <p id="loaderText" class="mt-4 text-xs text-slate-300" data-i18n="loader.starting">Preparando recortes…</p>
              <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div id="loaderBar" class="h-full w-0 rounded-full bg-teal-400 transition-all duration-300"></div>
              </div>
              <p class="mt-2 text-[10px] text-slate-500" data-i18n="loader.note">Los FITS se recortan y convierten en el servidor; no se guardan en disco.</p>
            </div>
          </div>

          <div id="emptyState" class="absolute inset-0 hidden place-items-center bg-[#05090f]/92 p-8 text-center">
            <div class="max-w-sm">
              <div class="mx-auto grid h-12 w-12 place-items-center rounded-full border border-slate-700 text-xl text-slate-500">◌</div>
              <h3 id="emptyTitle" class="mt-3 text-base font-semibold text-white" data-i18n="viewer.emptyTitle">Sin cobertura SPHEREx aquí</h3>
              <p id="emptyBody" class="mt-1 text-xs leading-relaxed text-slate-400" data-i18n="viewer.emptyBody">No hay imágenes calibradas DR1/QR3 que cubran estas coordenadas. Prueba otro objetivo.</p>
              <div class="mt-4 flex flex-wrap justify-center gap-2">
                <button id="retryBtn" type="button" class="chip rounded-lg px-3 py-2 text-xs text-slate-200 hover:border-teal-400/50" data-i18n="viewer.retry">Reintentar</button>
                <a id="emptyAladin" href="https://aladin.cds.unistra.fr/AladinLite/" target="_blank" rel="noopener" class="chip rounded-lg px-3 py-2 text-xs text-slate-200 hover:border-teal-400/50" data-i18n="viewer.checkAladin">Ver en Aladin</a>
              </div>
            </div>
          </div>

          <div class="absolute inset-x-0 bottom-0 h-1 bg-white/10">
            <div id="loadBar" class="h-full w-0 bg-teal-400 transition-all duration-300"></div>
          </div>
        </div>

        <section id="sciencePanel" class="sci-only panel fade-in rounded-2xl p-4">
          <div class="flex flex-wrap items-baseline justify-between gap-2">
            <h2 class="text-sm font-semibold text-white" data-i18n="sci.title">Datos científicos</h2>
            <span id="sciTarget" class="font-mono text-[10px] text-slate-500">—</span>
          </div>
          <p id="sciStatus" class="mt-1 text-[11px] leading-relaxed text-slate-400" data-i18n="sci.loading">Consultando Gaia DR3 y midiendo magnitudes…</p>

          <div class="mt-3 grid gap-2 sm:grid-cols-3">
            <div class="rounded-xl border border-[#21304a] bg-[#0B0F19]/60 p-3">
              <div class="text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="sci.distance">Distancia por paralaje</div>
              <div id="sciDistance" class="mt-1 font-mono text-lg text-teal-300">—</div>
              <div id="sciDistanceSub" class="mt-0.5 text-[10px] text-slate-500">—</div>
            </div>
            <div class="rounded-xl border border-[#21304a] bg-[#0B0F19]/60 p-3">
              <div class="text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="sci.parallax">Paralaje</div>
              <div id="sciParallax" class="mt-1 font-mono text-sm text-slate-100">—</div>
              <div id="sciParallaxSub" class="mt-0.5 text-[10px] text-slate-500">—</div>
            </div>
            <div class="rounded-xl border border-[#21304a] bg-[#0B0F19]/60 p-3">
              <div class="text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500" data-i18n="sci.source">Fuente Gaia DR3</div>
              <div id="sciSource" class="mt-1 break-all font-mono text-[11px] text-slate-100">—</div>
              <div id="sciSourceSub" class="mt-0.5 text-[10px] text-slate-500">—</div>
            </div>
          </div>

          <p class="mt-3 text-[10px] leading-relaxed text-slate-500" data-i18n="sci.note">La distancia es de la estrella Gaia DR3 más cercana al centro (radio 29″), no de la nebulosa o galaxia.</p>

          <div class="mt-4">
            <div class="flex flex-wrap items-baseline justify-between gap-2">
              <span class="text-xs font-semibold text-slate-200" data-i18n="sci.curve">Magnitud por fotograma</span>
              <span id="sciCurveInfo" class="text-[10px] text-slate-500">—</span>
            </div>
            <canvas id="curve" width="900" height="280" class="mt-2 block w-full rounded-xl border border-[#21304a] bg-[#080d16]" aria-label="Magnitude curve"></canvas>
            <p class="mt-2 text-[10px] leading-relaxed text-slate-500" data-i18n="sci.magNote">Magnitud AB de apertura (radio 1.8 px) medida sobre el recorte; no es una magnitud de catálogo.</p>
          </div>
        </section>

        <div class="panel rounded-2xl p-4">
          <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <button id="playBtn" type="button" class="btn-primary rounded-xl px-3 py-2 text-sm" data-i18n="ctrl.play">Reproducir</button>
            <button id="prevBtn" type="button" class="chip rounded-xl px-3 py-2 text-sm text-slate-200 hover:border-teal-400/50" data-i18n="ctrl.prev">◀ Anterior</button>
            <button id="nextBtn" type="button" class="chip rounded-xl px-3 py-2 text-sm text-slate-200 hover:border-teal-400/50" data-i18n="ctrl.next">Siguiente ▶</button>
            <label class="flex items-center gap-2 rounded-xl border border-[#21304a] bg-[#0B0F19]/60 px-3 py-2 text-xs text-slate-300">
              <span class="shrink-0 text-slate-500" data-i18n="ctrl.speed">Velocidad</span>
              <select id="speed" class="w-full bg-transparent text-xs text-white outline-none">
                <option value="1">1×</option>
                <option value="2">2×</option>
                <option value="5">5×</option>
              </select>
            </label>
          </div>

          <button id="blinkBtn" type="button" class="blink mt-3 w-full rounded-xl px-4 py-3 text-sm font-bold tracking-wide transition" data-i18n="ctrl.blink">Parpadeo rápido (primera vs última)</button>
          <p class="mt-1.5 text-center text-[10px] text-slate-500" data-i18n="ctrl.blinkNote">Alterna cada 400 ms entre la observación más antigua y la más reciente.</p>

          <div class="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-300">
            <label class="inline-flex cursor-pointer items-center gap-2">
              <input id="diffToggle" type="checkbox" class="h-4 w-4 accent-teal-400" />
              <span data-i18n="ctrl.diff">Resta (diferencia)</span>
            </label>
            <label class="inline-flex cursor-pointer items-center gap-2">
              <input id="circlesToggle" type="checkbox" class="h-4 w-4 accent-amber-400" checked />
              <span data-i18n="ctrl.circles">Rodear diferencias</span>
            </label>
            <span id="diffWarn" class="hidden text-[10px] text-amber-300/90" data-i18n="ctrl.diffWarnAll">Resta con todas las bandas: las diferencias pueden cubrir toda la imagen.</span>
            <label class="inline-flex cursor-pointer items-center gap-2">
              <input id="crosshairToggle" type="checkbox" class="h-4 w-4 accent-teal-400" checked />
              <span data-i18n="ctrl.crosshair">Retícula</span>
            </label>
            <span id="autoBandNote" class="sci-only text-[10px] text-teal-300/80"></span>
          </div>

          <div class="mt-4">
            <div class="flex items-center justify-between text-[10px] text-slate-500">
              <span data-i18n="ctrl.timeline">Línea de tiempo</span>
              <span id="frameInfo">—</span>
            </div>
            <input id="timeline" type="range" min="0" max="0" value="0" class="mt-2 w-full" aria-label="Timeline" />
          </div>

          <div class="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
            <span class="text-slate-500" data-i18n="ctrl.zoom">Zoom</span>
            <button id="zoomOut" type="button" class="zoom-btn" aria-label="Zoom out">−</button>
            <span id="zoomLevel" class="w-10 text-center font-mono text-slate-200">1×</span>
            <button id="zoomIn" type="button" class="zoom-btn" aria-label="Zoom in">+</button>
            <button id="zoomReset" type="button" class="chip rounded-lg px-2 py-1 text-[10px] text-slate-300 hover:border-teal-400/50 disabled:opacity-40" data-i18n="ctrl.zoomReset">Reiniciar</button>
            <span class="text-[10px] text-slate-500" data-i18n="ctrl.zoomHint">Rueda para zoom · arrastra para mover</span>
          </div>
        </div>

        <div class="panel rounded-2xl p-4">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <h2 class="text-sm font-semibold text-white" data-i18n="actions.title">Compartir y verificar</h2>
            <span class="text-[10px] text-slate-500" data-i18n="actions.hint">Comprueba si ya está en WISE/Gaia antes de reportar un candidato.</span>
          </div>
          <div class="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <button id="shareBtn" type="button" class="chip rounded-xl px-3 py-2 text-xs text-slate-200 hover:border-teal-400/50" data-i18n="actions.share">Compartir coordenadas</button>
            <button id="gifBtn" type="button" class="chip rounded-xl px-3 py-2 text-xs text-slate-200 hover:border-teal-400/50" data-i18n="actions.gif">Exportar GIF</button>
            <a id="aladin" href="https://aladin.cds.unistra.fr/AladinLite/" target="_blank" rel="noopener" class="chip rounded-xl px-3 py-2 text-center text-xs text-slate-200 hover:border-teal-400/50" data-i18n="actions.aladin">Aladin Lite</a>
            <a id="finder" href="https://irsa.ipac.caltech.edu/applications/finderchart/" target="_blank" rel="noopener" class="chip rounded-xl px-3 py-2 text-center text-xs text-slate-200 hover:border-teal-400/50" data-i18n="actions.finder">IRSA Finder Chart</a>
          </div>
          <a id="vizier" href="https://vizier.cds.unistra.fr/" target="_blank" rel="noopener" class="mt-2 inline-block text-[11px] text-teal-300/90 underline decoration-teal-300/30 underline-offset-2 hover:text-teal-200" data-i18n="actions.vizier">Buscar en VizieR (WISE / Gaia)</a>
        </div>
      </section>
    </main>

    <footer class="mx-auto max-w-7xl px-4 pb-8 pt-2 text-[10px] leading-relaxed text-slate-600">
      <p><span data-i18n="footer.note">Datos públicos de SPHEREx (NASA/IPAC IRSA, DR1/QR3). Los recortes se generan al vuelo y se sirven como PNG; no se almacenan FITS en disco.</span></p>
      <p class="mt-1" data-i18n="footer.built">Prototipo para NASA Space Apps Challenge 2026 · Reto «Planeta X y SPHEREx» · Detección no validada.</p>
    </footer>

    <div id="toast" class="pointer-events-none fixed bottom-5 left-1/2 z-50 hidden -translate-x-1/2 rounded-xl border border-teal-400/40 bg-[#0B0F19]/95 px-4 py-2 text-xs text-teal-200 shadow-2xl"></div>

    <script>
      // ------------------------------------------------------------------ i18n
      const I18N = {
        es: {
          "app.subtitle": "Time-lapse de recortes infrarrojos reales · NASA/IPAC IRSA",
          "nav.api": "API docs",
          "search.title": "Buscar en el cielo",
          "search.hint": "Escribe un nombre (M42, Betelgeuse) o coordenadas ICRS/J2000 en grados decimales. Se consultan las colecciones calibradas spherex_dr1 y spherex_qr3.",
          "search.name": "Nombre del objeto",
          "search.namePlaceholder": "M42, Betelgeuse, M63…",
          "search.resolve": "Resolver",
          "search.resolved": "{name} resuelto por {resolver}.",
          "search.ra": "RA (°)",
          "search.dec": "Dec (°)",
          "search.fov": "FOV",
          "search.band": "Banda",
          "search.colormap": "Mapa de color",
          "search.submit": "Buscar observaciones",
          "band.auto": "Auto (banda con más épocas)",
          "band.all": "Todas las bandas",
          "band.option": "D{n} · {l} µm ({count})",
          "cmap.gray": "Escala de grises",
          "cmap.inferno": "Inferno",
          "cmap.viridis": "Viridis",
          "presets.title": "Objetivos de prueba",
          "presets.hint": "Un clic busca",
          "preset.barnard.name": "Estrella de Barnard",
          "preset.barnard.note": "Alto movimiento propio",
          "preset.epsindi.name": "Epsilon Indi B",
          "preset.epsindi.note": "Enana marrón cercana",
          "preset.wise0855.name": "WISE 0855",
          "preset.wise0855.note": "Enana marrón ultra fría",
          "preset.rwaur.name": "RW Aurigae",
          "preset.rwaur.note": "Estrella variable",
          "preset.rdra.name": "R Draconis",
          "preset.rdra.note": "Variable Mira",
          "preset.m1.name": "M1 (Nebulosa del Cangrejo)",
          "preset.m1.note": "Resto de supernova",
          "preset.orion.name": "Nebulosa de Orión",
          "preset.orion.note": "Región de formación estelar",
          "preset.m63.name": "M63 (Galaxia del Girasol)",
          "preset.m63.note": "Galaxia espiral",
          "preset.m57.name": "M57 (Nebulosa del Anillo)",
          "preset.m57.note": "Nebulosa planetaria",
          "tips.title": "Cómo leer el visor",
          "tips.one": "Usa Parpadeo rápido para alternar la primera y la última fecha: un punto que salta es un candidato a objeto móvil.",
          "tips.two": "Activa Resta para restar el primer fotograma: las estrellas fijas quedan en gris y el movimiento aparece como dipolo blanco/negro.",
          "tips.three": "Filtra por banda D1–D6 para evitar parpadeos de brillo entre filtros infrarrojos.",
          "tips.four": "Esto es un explorador, no una detección validada: verifica siempre en Aladin/VizieR.",
          "hud.mission": "Misión",
          "hud.wavelength": "Longitud de onda",
          "hud.fov": "Campo",
          "hud.coords": "Coordenadas",
          "hud.diffLabel": "Modo",
          "hud.diff": "RESTA",
          "hud.regions": "Círculos: {n}",
          "hud.noRegions": "Sin diferencias claras",
          "loader.starting": "Preparando recortes…",
          "loader.note": "Los FITS se recortan y convierten en el servidor; no se guardan en disco.",
          "loader.progress": "Descargando fotogramas: {done} / {total} listos",
          "viewer.emptyTitle": "Sin cobertura SPHEREx aquí",
          "viewer.emptyBody": "No hay imágenes calibradas DR1/QR3 que cubran estas coordenadas. Prueba otro objetivo o un campo mayor.",
          "viewer.retry": "Reintentar",
          "viewer.checkAladin": "Ver en Aladin",
          "viewer.errorTitle": "No se pudo cargar la secuencia",
          "viewer.errorBody": "IRSA no respondió o devolvió un error. Revisa la conexión y vuelve a intentarlo.",
          "ctrl.play": "Reproducir",
          "ctrl.pause": "Pausar",
          "ctrl.prev": "◀ Anterior",
          "ctrl.next": "Siguiente ▶",
          "ctrl.speed": "Velocidad",
          "ctrl.blink": "Parpadeo rápido (primera vs última)",
          "ctrl.blinkOn": "Detener parpadeo",
          "ctrl.blinkNote": "Alterna cada 400 ms entre la observación más antigua y la más reciente.",
          "ctrl.diff": "Resta (diferencia)",
          "ctrl.circles": "Rodear diferencias",
          "ctrl.diffWarnAll": "Resta con todas las bandas: las diferencias pueden cubrir toda la imagen.",
          "ctrl.crosshair": "Retícula",
          "ctrl.zoom": "Zoom",
          "ctrl.zoomReset": "Reiniciar",
          "ctrl.zoomHint": "Rueda para zoom · arrastra para mover",
          "ctrl.timeline": "Línea de tiempo",
          "ctrl.frameInfo": "{i} / {n} · {date}",
          "actions.title": "Compartir y verificar",
          "actions.hint": "Comprueba si ya está en WISE/Gaia antes de reportar un candidato.",
          "actions.share": "Compartir estado",
          "actions.gif": "Exportar GIF",
          "actions.aladin": "Aladin Lite",
          "actions.finder": "IRSA Finder Chart",
          "actions.vizier": "Buscar en VizieR (WISE / Gaia)",
          "status.idle": "Elige un objetivo o introduce coordenadas.",
          "status.searching": "Consultando IRSA (SIA v2)…",
          "status.retrying": "IRSA no responde; reintentando ({n}/{max})…",
          "status.resolving": "Resolviendo el nombre…",
          "status.found": "{n} fotogramas en {bands} banda(s).",
          "status.sampled": "Muestra: {n} de {total} fotogramas en {bands} bandas.",
          "status.bandFrames": "{n} fotogramas en la banda D{band}.",
          "status.sampledBand": "Muestra: {n} de {total} fotogramas de la banda D{band}.",
          "status.autoBand": "Banda automática: D{band}.",
          "status.emptyBand": "No hay fotogramas de esta banda; prueba «Todas».",
          "status.empty": "Sin cobertura de SPHEREx en estas coordenadas.",
          "status.imageErrors": "{n} fotograma(s) no se pudieron descargar.",
          "error.ra": "RA debe ser un número entre 0 y 360.",
          "error.dec": "Dec debe ser un número entre −90 y 90.",
          "error.name": "Escribe un nombre de objeto o usa coordenadas.",
          "error.resolve": "No se pudo resolver el nombre. Revisa la conexión o usa coordenadas.",
          "error.resolveNotFound": "No se encontró «{name}». Prueba otro nombre o coordenadas.",
          "error.network": "No se pudo conectar con el servidor local. ¿Está levantado el backend?",
          "error.observations": "IRSA no respondió a la búsqueda. Inténtalo de nuevo en unos segundos.",
          "error.upstream": "El backend local funciona, pero IRSA (Caltech) no respondió. Inténtalo de nuevo en unos segundos.",
          "error.detail": "Detalle: {detail}",
          "actions.copied": "Enlace copiado al portapapeles.",
          "actions.copyfail": "No se pudo copiar. URL: {url}",
          "actions.gifBusy": "Generando GIF… puede tardar.",
          "actions.gifDone": "GIF descargado.",
          "actions.gifFail": "No se pudo generar el GIF (demasiados fotogramas o IRSA no respondió).",
          "actions.gifFew": "Necesitas al menos 2 fotogramas para exportar.",
          "blink.needTwo": "Hacen falta al menos 2 fotogramas para el parpadeo.",
          "mode.casual": "Casual",
          "mode.science": "Científico",
          "sci.title": "Datos científicos",
          "sci.distance": "Distancia por paralaje",
          "sci.parallax": "Paralaje",
          "sci.source": "Fuente Gaia DR3",
          "sci.loading": "Consultando Gaia DR3 y midiendo magnitudes…",
          "sci.unavailable": "Gaia DR3 no respondió. Inténtalo de nuevo.",
          "sci.noSource": "No hay estrellas Gaia DR3 en 29″.",
          "sci.ly": "{v} años luz",
          "sci.pc": "{v} pc",
          "sci.range": "rango {low}–{high} pc",
          "sci.parallaxValue": "{v} mas · error {e} mas",
          "sci.reliable": "medición fiable",
          "sci.unreliable": "medición poco fiable",
          "sci.sourceSub": "G {g} · {sep}″ del centro",
          "sci.curve": "Magnitud por fotograma",
          "sci.curvePoints": "{n} de {total} fotogramas con magnitud",
          "sci.curveEmpty": "Sin magnitudes válidas para esta secuencia.",
          "sci.magNote": "Magnitud AB de apertura (radio 1.8 px) medida sobre el recorte; no es una magnitud de catálogo.",
          "sci.note": "La distancia es de la estrella Gaia DR3 más cercana al centro (radio 29″), no de la nebulosa o galaxia.",
          "footer.note": "Datos públicos de SPHEREx (NASA/IPAC IRSA, DR1/QR3). Los recortes se generan al vuelo y se sirven como PNG; no se almacenan FITS en disco.",
          "footer.built": "Prototipo para NASA Space Apps Challenge 2026 · Reto «Planeta X y SPHEREx» · Detección no validada.",
        },
        en: {
          "app.subtitle": "Time-lapse of real infrared cutouts · NASA/IPAC IRSA",
          "nav.api": "API docs",
          "search.title": "Search the sky",
          "search.hint": "Type a name (M42, Betelgeuse) or ICRS/J2000 coordinates in decimal degrees. Queries the calibrated spherex_dr1 and spherex_qr3 collections.",
          "search.name": "Object name",
          "search.namePlaceholder": "M42, Betelgeuse, M63…",
          "search.resolve": "Resolve",
          "search.resolved": "{name} resolved by {resolver}.",
          "search.ra": "RA (°)",
          "search.dec": "Dec (°)",
          "search.fov": "FOV",
          "search.band": "Band",
          "search.colormap": "Colormap",
          "search.submit": "Search observations",
          "band.auto": "Auto (band with most epochs)",
          "band.all": "All bands",
          "band.option": "D{n} · {l} µm ({count})",
          "cmap.gray": "Grayscale",
          "cmap.inferno": "Inferno",
          "cmap.viridis": "Viridis",
          "presets.title": "Test targets",
          "presets.hint": "One click searches",
          "preset.barnard.name": "Barnard's Star",
          "preset.barnard.note": "High proper motion",
          "preset.epsindi.name": "Epsilon Indi B",
          "preset.epsindi.note": "Nearby brown dwarf",
          "preset.wise0855.name": "WISE 0855",
          "preset.wise0855.note": "Ultra-cool brown dwarf",
          "preset.rwaur.name": "RW Aurigae",
          "preset.rwaur.note": "Variable star",
          "preset.rdra.name": "R Draconis",
          "preset.rdra.note": "Mira variable",
          "preset.m1.name": "M1 (Crab Nebula)",
          "preset.m1.note": "Supernova remnant",
          "preset.orion.name": "Orion Nebula",
          "preset.orion.note": "Star-forming region",
          "preset.m63.name": "M63 (Sunflower Galaxy)",
          "preset.m63.note": "Spiral galaxy",
          "preset.m57.name": "M57 (Ring Nebula)",
          "preset.m57.note": "Planetary nebula",
          "tips.title": "How to read the viewer",
          "tips.one": "Use rapid blink to alternate the first and last dates: a dot that jumps is a moving-object candidate.",
          "tips.two": "Turn on Difference to subtract the first frame: fixed stars go grey and motion appears as a white/black dipole.",
          "tips.three": "Filter by band D1–D6 to avoid brightness flicker between infrared filters.",
          "tips.four": "This is an explorer, not a validated detection: always verify in Aladin/VizieR.",
          "hud.mission": "Mission",
          "hud.wavelength": "Wavelength",
          "hud.fov": "Field",
          "hud.coords": "Coordinates",
          "hud.diffLabel": "Mode",
          "hud.diff": "DIFF",
          "hud.regions": "Circles: {n}",
          "hud.noRegions": "No clear differences",
          "loader.starting": "Preparing cutouts…",
          "loader.note": "FITS files are cut and rendered on the server; nothing is stored on disk.",
          "loader.progress": "Downloading frames: {done} / {total} ready",
          "viewer.emptyTitle": "No SPHEREx coverage here",
          "viewer.emptyBody": "No calibrated DR1/QR3 images cover these coordinates. Try another target or a larger field.",
          "viewer.retry": "Retry",
          "viewer.checkAladin": "Check in Aladin",
          "viewer.errorTitle": "Could not load the sequence",
          "viewer.errorBody": "IRSA did not respond or returned an error. Check your connection and try again.",
          "ctrl.play": "Play",
          "ctrl.pause": "Pause",
          "ctrl.prev": "◀ Previous",
          "ctrl.next": "Next ▶",
          "ctrl.speed": "Speed",
          "ctrl.blink": "Rapid blink (first vs last)",
          "ctrl.blinkOn": "Stop blinking",
          "ctrl.blinkNote": "Alternates every 400 ms between the oldest and newest observation.",
          "ctrl.diff": "Difference (subtract first)",
          "ctrl.circles": "Circle differences",
          "ctrl.diffWarnAll": "Difference across all bands: changes may cover the whole frame.",
          "ctrl.crosshair": "Crosshair",
          "ctrl.zoom": "Zoom",
          "ctrl.zoomReset": "Reset",
          "ctrl.zoomHint": "Wheel to zoom · drag to pan",
          "ctrl.timeline": "Timeline",
          "ctrl.frameInfo": "{i} / {n} · {date}",
          "actions.title": "Share and verify",
          "actions.hint": "Check WISE/Gaia before reporting a candidate.",
          "actions.share": "Share state",
          "actions.gif": "Export GIF",
          "actions.aladin": "Aladin Lite",
          "actions.finder": "IRSA Finder Chart",
          "actions.vizier": "Search VizieR (WISE / Gaia)",
          "status.idle": "Pick a target or enter coordinates.",
          "status.searching": "Querying IRSA (SIA v2)…",
          "status.retrying": "IRSA is not answering; retrying ({n}/{max})…",
          "status.resolving": "Resolving the name…",
          "status.found": "{n} frames across {bands} band(s).",
          "status.sampled": "Sample: {n} of {total} frames across {bands} bands.",
          "status.bandFrames": "{n} frames in band D{band}.",
          "status.sampledBand": "Sample: {n} of {total} frames in band D{band}.",
          "status.autoBand": "Automatic band: D{band}.",
          "status.emptyBand": "No frames in this band; try “All”.",
          "status.empty": "No SPHEREx coverage at these coordinates.",
          "status.imageErrors": "{n} frame(s) could not be downloaded.",
          "error.ra": "RA must be a number between 0 and 360.",
          "error.dec": "Dec must be a number between −90 and 90.",
          "error.name": "Enter an object name or use coordinates.",
          "error.resolve": "Could not resolve the name. Check the connection or use coordinates.",
          "error.resolveNotFound": "No object found for “{name}”. Try another name or coordinates.",
          "error.network": "Could not reach the local server. Is the backend running?",
          "error.observations": "IRSA did not answer the search. Try again in a few seconds.",
          "error.upstream": "The local backend is running, but IRSA (Caltech) did not respond. Try again in a few seconds.",
          "error.detail": "Detail: {detail}",
          "actions.copied": "Link copied to clipboard.",
          "actions.copyfail": "Could not copy. URL: {url}",
          "actions.gifBusy": "Building GIF… this may take a while.",
          "actions.gifDone": "GIF downloaded.",
          "actions.gifFail": "Could not build the GIF (too many frames or IRSA did not respond).",
          "actions.gifFew": "You need at least 2 frames to export.",
          "blink.needTwo": "At least 2 frames are needed to blink.",
          "mode.casual": "Casual",
          "mode.science": "Science",
          "sci.title": "Scientific data",
          "sci.distance": "Parallax distance",
          "sci.parallax": "Parallax",
          "sci.source": "Gaia DR3 source",
          "sci.loading": "Querying Gaia DR3 and measuring magnitudes…",
          "sci.unavailable": "Gaia DR3 did not respond. Try again.",
          "sci.noSource": "No Gaia DR3 stars within 29″.",
          "sci.ly": "{v} light-years",
          "sci.pc": "{v} pc",
          "sci.range": "range {low}–{high} pc",
          "sci.parallaxValue": "{v} mas · error {e} mas",
          "sci.reliable": "reliable measurement",
          "sci.unreliable": "low-confidence measurement",
          "sci.sourceSub": "G {g} · {sep}″ from centre",
          "sci.curve": "Magnitude per frame",
          "sci.curvePoints": "{n} of {total} frames with magnitude",
          "sci.curveEmpty": "No valid magnitudes for this sequence.",
          "sci.magNote": "Aperture AB magnitude (1.8 px radius) measured on the cutout; this is not a catalogue magnitude.",
          "sci.note": "The distance is for the nearest Gaia DR3 star to the centre (29″ radius), not for the nebula or galaxy.",
          "footer.note": "Public SPHEREx data (NASA/IPAC IRSA, DR1/QR3). Cutouts are rendered on the fly and served as PNG; no FITS files are stored on disk.",
          "footer.built": "Prototype for NASA Space Apps Challenge 2026 · “Planet X & SPHEREx” challenge · Unvalidated detection.",
        },
      };

      let LANG = "es";
      try {
        const saved = localStorage.getItem("pxs-lang");
        if (saved === "es" || saved === "en") LANG = saved;
      } catch {}

      const t = (key, vars) => {
        let text = I18N[LANG]?.[key] ?? I18N.es[key] ?? key;
        if (vars) for (const [k, v] of Object.entries(vars)) text = text.replaceAll(`{${k}}`, String(v));
        return text;
      };
      const $ = (id) => document.getElementById(id);

      // ----------------------------------------------------------------- state
      const PRESETS = [
        { id: "barnard", ra: 269.452, dec: 4.693 },
        { id: "epsindi", ra: 330.84, dec: -56.786 },
        { id: "wise0855", ra: 133.786, dec: -7.244 },
        { id: "rwaur", ra: 76.956, dec: 30.401 },
        { id: "rdra", ra: 248.198, dec: 66.743 },
        { id: "m1", ra: 83.6331, dec: 22.0145 },
        { id: "orion", ra: 83.8221, dec: -5.3911 },
        { id: "m63", ra: 198.9553, dec: 42.0294 },
        { id: "m57", ra: 283.3963, dec: 33.0292 },
      ];

      const S = {
        ra: 269.452,
        dec: 4.693,
        fov: 0.05,
        cmap: "gray",
        mode: "casual",
        science: null,
        photo: null,
        scienceID: 0,
        allFrames: [],
        frames: [],
        bands: [],
        images: [],
        index: 0,
        playing: false,
        blinking: false,
        diff: false,
        circles: true,
        speed: 1,
        playTimer: null,
        blinkTimer: null,
        loading: false,
        loaded: 0,
        failed: 0,
        started: false,
        total: 0,
        truncated: false,
        allTotal: 0,
        allTruncated: false,
        diffCache: new Map(),
        requestID: 0,
        filterID: 0,
        loadID: 0,
        pendingBand: null,
        cross: true,
        zoom: 1,
        cx: 0.5,
        cy: 0.5,
        pendingFrame: null,
        object: "",
        status: { key: "status.idle", vars: null, tone: "muted" },
        error: null,
      };

      const canvas = $("view");
      const ctx = canvas.getContext("2d");

      // ------------------------------------------------------------- utilities
      function showToast(message) {
        const toast = $("toast");
        toast.textContent = message;
        toast.classList.remove("hidden");
        toast.classList.add("fade-in");
        clearTimeout(showToast.timer);
        showToast.timer = setTimeout(() => toast.classList.add("hidden"), 4200);
      }

      function setStatus(key, vars = null, tone = "muted") {
        S.status = { key, vars, tone };
        renderStatus();
      }

      function renderStatus() {
        const el = $("status");
        el.textContent = t(S.status.key, S.status.vars);
        el.className =
          "mt-3 rounded-lg border px-3 py-2 text-[11px] leading-relaxed " +
          (S.status.tone === "error"
            ? "border-red-500/30 bg-red-500/10 text-red-300"
            : S.status.tone === "ok"
            ? "border-teal-500/25 bg-teal-500/5 text-teal-200/90"
            : "border-[#21304a] bg-[#0B0F19]/60 text-slate-400");
      }

      function showError(key, detail = "", vars = null) {
        S.error = { key, detail, vars };
        renderError();
      }

      function renderError() {
        const el = $("error");
        if (!S.error) {
          el.classList.add("hidden");
          el.textContent = "";
          return;
        }
        let text = t(S.error.key, S.error.vars);
        if (S.error.detail) text += " " + t("error.detail", { detail: S.error.detail });
        el.textContent = text;
        el.classList.remove("hidden");
      }

      function clearError() {
        S.error = null;
        renderError();
      }

      function setBusy(busy) {
        const btn = $("searchBtn");
        btn.disabled = busy;
        btn.classList.toggle("cursor-wait", busy);
        btn.innerHTML = busy
          ? `<span class="inline-flex items-center gap-2"><span class="h-3 w-3 animate-spin rounded-full border border-[#052e2b]/30 border-t-[#052e2b]"></span>${t("status.searching")}</span>`
          : `<span>${t("search.submit")}</span>`;
      }

      function fmtDate(iso) {
        if (!iso) return "—";
        return iso.replace("T", " ").replace("Z", " UTC");
      }

      function frameUrl(frame, fov, cmap) {
        const params = new URLSearchParams({
          fits_url: frame.access_url,
          ra: String(S.ra),
          dec: String(S.dec),
          size: String(fov),
          colormap: cmap,
        });
        return `/api/cutout?${params.toString()}`;
      }

      async function getJSON(url, timeout = 65000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
          let response;
          try {
            response = await fetch(url, { signal: controller.signal });
          } catch (cause) {
            const error = new Error(controller.signal.aborted ? "Request timed out" : "Network request failed");
            error.isAbort = controller.signal.aborted;
            error.isNetwork = !controller.signal.aborted;
            throw error;
          }
          const data = await response.json().catch(() => ({}));
          if (!response.ok) {
            const detail = data.detail || `HTTP ${response.status}`;
            const error = new Error(detail);
            error.status = response.status;
            error.detail = detail;
            throw error;
          }
          return data;
        } finally {
          clearTimeout(timer);
        }
      }

      async function postJSON(url, body, timeout = 120000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
          let response;
          try {
            response = await fetch(url, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
              signal: controller.signal,
            });
          } catch {
            const error = new Error(controller.signal.aborted ? "Request timed out" : "Network request failed");
            error.isAbort = controller.signal.aborted;
            error.isNetwork = !controller.signal.aborted;
            throw error;
          }
          const data = await response.json().catch(() => ({}));
          if (!response.ok) {
            const detail = data.detail || `HTTP ${response.status}`;
            const error = new Error(detail);
            error.status = response.status;
            error.detail = detail;
            throw error;
          }
          return data;
        } finally {
          clearTimeout(timer);
        }
      }

      // ---------------------------------------------------------------- i18n
      function applyStatic() {
        document.documentElement.lang = LANG;
        document.querySelectorAll("[data-i18n]").forEach((node) => {
          node.textContent = t(node.dataset.i18n);
        });
        document.querySelectorAll("[data-i18n-html]").forEach((node) => {
          node.innerHTML = t(node.dataset.i18nHtml);
        });
        document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
          node.placeholder = t(node.dataset.i18nPlaceholder);
        });
        document.querySelectorAll("#lang button").forEach((node) => {
          const active = node.dataset.lang === LANG;
          node.classList.toggle("bg-[#182238]", active);
          node.classList.toggle("text-teal-300", active);
          node.classList.toggle("text-slate-400", !active);
        });
      }

      function renderPresets() {
        const wrap = $("presets");
        wrap.innerHTML = "";
        for (const preset of PRESETS) {
          const button = document.createElement("button");
          button.type = "button";
          button.className =
            "preset rounded-xl border border-[#21304a] bg-[#0B0F19]/50 px-3 py-2 text-left transition";
          button.innerHTML = `
            <span class="block text-xs font-semibold text-slate-100">${t(`preset.${preset.id}.name`)}</span>
            <span class="mt-0.5 block font-mono text-[10px] text-slate-500">RA ${preset.ra.toFixed(3)} · Dec ${preset.dec >= 0 ? "+" : ""}${preset.dec.toFixed(3)}</span>
            <span class="mt-0.5 block text-[10px] text-teal-300/80">${t(`preset.${preset.id}.note`)}</span>`;
          button.addEventListener("click", () => {
            $("ra").value = preset.ra;
            $("dec").value = preset.dec;
            S.object = "";
            $("objectName").value = "";
            document.querySelectorAll("#presets .preset").forEach((node) => node.classList.remove("active"));
            button.classList.add("active");
            search();
          });
          wrap.appendChild(button);
        }
      }

      function rebuildBandOptions() {
        const select = $("band");
        const previous = S.pendingBand ?? select.value;
        const options = [
          `<option value="auto">${t("band.auto")}</option>`,
          `<option value="all">${t("band.all")}</option>`,
          ...S.bands.map((band) => {
            const center = ((band.min_um + band.max_um) / 2).toFixed(2);
            return `<option value="${band.band}">${t("band.option", { n: band.band, l: center, count: band.count })}</option>`;
          }),
        ];
        select.innerHTML = options.join("");
        if ([...select.options].some((option) => option.value === previous)) select.value = previous;
        else select.value = "auto";
        S.pendingBand = null;
      }

      function setLang(next) {
        LANG = next;
        try {
          localStorage.setItem("pxs-lang", LANG);
        } catch {}
        applyStatic();
        renderPresets();
        rebuildBandOptions();
        renderStatus();
        renderError();
        updateControlsLabels();
        updatePlayButton(!S.playing);
        updateHud();
        updateFrameInfo();
        updateLinks();
        updateZoomUI();
        updateCrosshair();
        if (S.frames.length) setStatus(S.status.key, S.status.vars, S.status.tone);
        renderScience();
        render();
      }

      // --------------------------------------------------------------- presets
      function setCoords(ra, dec) {
        S.ra = ra;
        S.dec = dec;
        $("ra").value = ra;
        $("dec").value = dec;
      }

      function syncUrl() {
        const params = new URLSearchParams({
          ra: S.ra.toFixed(6),
          dec: S.dec.toFixed(6),
          fov: String(S.fov),
          band: $("band").value || "auto",
          cmap: S.cmap,
          frame: String(S.index),
          diff: S.diff ? "1" : "0",
          circles: S.circles ? "1" : "0",
          cross: S.cross ? "1" : "0",
          speed: String(S.speed),
          z: S.zoom.toFixed(2),
          cx: S.cx.toFixed(4),
          cy: S.cy.toFixed(4),
          mode: S.mode,
          lang: LANG,
        });
        if (S.object) params.set("object", S.object);
        try {
          history.replaceState(null, "", `?${params.toString()}`);
        } catch {}
      }

      let urlTimer = null;

      function scheduleSyncUrl(delay = 180) {
        clearTimeout(urlTimer);
        urlTimer = setTimeout(syncUrl, delay);
      }

      function updateLinks() {
        const ra = S.ra.toFixed(3);
        const dec = S.dec.toFixed(3);
        const arcmin = Math.max(0.5, S.fov * 60).toFixed(1);
        $("aladin").href = `https://aladin.cds.unistra.fr/AladinLite/?target=${ra}%20${dec}&fov=${arcmin}`;
        $("emptyAladin").href = $("aladin").href;
        $("finder").href = `https://irsa.ipac.caltech.edu/applications/finderchart/?RA=${ra}&DEC=${dec}`;
        $("vizier").href = `https://vizier.cds.unistra.fr/viz-bin/VizieR?-c=${ra}%20${dec}&-c.rs=${arcmin}`;
      }

      // ---------------------------------------------------------------- modes
      function setMode(next, sync = true) {
        S.mode = next === "science" ? "science" : "casual";
        document.body.dataset.mode = S.mode;
        document.querySelectorAll("#modeSwitch button").forEach((node) => {
          const active = node.dataset.mode === S.mode;
          node.classList.toggle("active", active);
          node.setAttribute("aria-pressed", String(active));
        });
        try {
          localStorage.setItem("pxs-mode", S.mode);
        } catch {}
        if (sync) scheduleSyncUrl();
        if (S.mode === "science") loadScience();
        else {
          S.scienceID++;
          drawCurve();
        }
      }

      function fmtNumber(value) {
        if (!Number.isFinite(value)) return "—";
        return Math.round(value).toLocaleString(LANG === "es" ? "es-ES" : "en-US");
      }

      function sampleFrames(frames, limit) {
        if (frames.length <= limit) return frames.slice();
        const step = (frames.length - 1) / (limit - 1);
        return Array.from({ length: limit }, (_, i) => frames[Math.round(i * step)]);
      }

      async function loadScience() {
        if (!S.frames.length) {
          S.science = null;
          S.photo = null;
          renderScience();
          return;
        }
        const id = ++S.scienceID;
        S.science = null;
        S.photo = null;
        renderScience();
        const ra = S.ra;
        const dec = S.dec;
        const body = {
          frames: sampleFrames(S.frames, 12).map((frame) => ({
            fits_url: frame.access_url,
            date: frame.date,
            band: frame.band,
          })),
          ra,
          dec,
          size: S.fov,
        };
        const [science, photo] = await Promise.all([
          getJSON(`/api/science?ra=${ra}&dec=${dec}`, 45000).catch(() => null),
          postJSON("/api/photometry", body, 240000).catch(() => null),
        ]);
        if (id !== S.scienceID) return;
        S.science = science;
        S.photo = photo;
        renderScience();
      }

      function renderScience() {
        const science = S.science;
        const sign = S.dec >= 0 ? "+" : "";
        $("sciTarget").textContent = `${S.object || "RA/Dec"} · ${S.ra.toFixed(3)}°, ${sign}${S.dec.toFixed(3)}°`;

        const status = $("sciStatus");
        if (!S.frames.length) {
          const empty = S.status.key === "status.empty" || S.status.key === "status.emptyBand";
          status.textContent = empty ? t("sci.unavailable") : t("sci.loading");
          status.classList.remove("hidden");
        } else if (!science) {
          status.textContent = t("sci.loading");
          status.classList.remove("hidden");
        } else if (!science.available) {
          status.textContent = science.reason === "no_source" ? t("sci.noSource") : t("sci.unavailable");
          status.classList.remove("hidden");
        } else {
          status.classList.add("hidden");
        }

        if (science?.available && science.distance_pc) {
          $("sciDistance").textContent = t("sci.ly", { v: fmtNumber(science.distance_ly) });
          let sub = t("sci.pc", { v: fmtNumber(science.distance_pc) });
          if (science.distance_pc_low && science.distance_pc_high) {
            sub += ` · ${t("sci.range", {
              low: fmtNumber(science.distance_pc_low),
              high: fmtNumber(science.distance_pc_high),
            })}`;
          }
          $("sciDistanceSub").textContent = sub;
        } else {
          $("sciDistance").textContent = "—";
          $("sciDistanceSub").textContent = "—";
        }

        if (science?.available && science.parallax_mas != null) {
          $("sciParallax").textContent = t("sci.parallaxValue", {
            v: science.parallax_mas.toFixed(3),
            e: science.parallax_error_mas != null ? science.parallax_error_mas.toFixed(3) : "—",
          });
          const bits = [];
          if (science.snr) bits.push(`SNR ${science.snr}`);
          bits.push(science.reliable ? t("sci.reliable") : t("sci.unreliable"));
          $("sciParallaxSub").textContent = bits.join(" · ");
          $("sciParallaxSub").className = `mt-0.5 text-[10px] ${
            science.reliable ? "text-teal-300/80" : "text-amber-300/80"
          }`;
        } else {
          $("sciParallax").textContent = "—";
          $("sciParallaxSub").textContent = "—";
          $("sciParallaxSub").className = "mt-0.5 text-[10px] text-slate-500";
        }

        if (science?.available) {
          $("sciSource").textContent = `Gaia DR3 #${science.source_id}`;
          $("sciSourceSub").textContent = t("sci.sourceSub", {
            g: science.g_mag != null ? science.g_mag.toFixed(2) : "—",
            sep: science.separation_arcsec != null ? science.separation_arcsec.toFixed(2) : "—",
          });
        } else {
          $("sciSource").textContent = "—";
          $("sciSourceSub").textContent = "—";
        }

        drawCurve();
      }

      function drawCurve() {
        const chart = $("curve");
        if (!chart) return;
        const g = chart.getContext("2d");
        const width = chart.width;
        const height = chart.height;
        g.clearRect(0, 0, width, height);
        g.fillStyle = "#080d16";
        g.fillRect(0, 0, width, height);

        const points = (S.photo?.points || [])
          .filter(
            (point) =>
              Number.isFinite(point.mag) &&
              (!Number.isFinite(point.mag_error) || point.mag_error <= 1.5)
          )
          .slice()
          .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
        const total = S.photo?.points?.length || 0;
        $("sciCurveInfo").textContent = S.photo ? t("sci.curvePoints", { n: points.length, total }) : "—";

        if (points.length < 2) {
          g.fillStyle = "#475569";
          g.font = "16px ui-monospace, monospace";
          g.textAlign = "center";
          g.textBaseline = "middle";
          g.fillText(t("sci.curveEmpty"), width / 2, height / 2);
          return;
        }

        const pad = { left: 72, right: 20, top: 20, bottom: 44 };
        const errors = points.map((point) =>
          Number.isFinite(point.mag_error) ? Math.max(0, point.mag_error) : 0
        );
        let low = Math.min(...points.map((point, i) => point.mag - errors[i]));
        let high = Math.max(...points.map((point, i) => point.mag + errors[i]));
        if (!Number.isFinite(low) || !Number.isFinite(high)) return;
        if (high - low < 0.2) {
          low -= 0.1;
          high += 0.1;
        }
        const padding = (high - low) * 0.12;
        low -= padding;
        high += padding;

        const xAt = (i) => pad.left + (i / (points.length - 1)) * (width - pad.left - pad.right);
        const yAt = (mag) => pad.top + ((mag - low) / (high - low)) * (height - pad.top - pad.bottom);

        g.strokeStyle = "#1e293b";
        g.fillStyle = "#64748b";
        g.lineWidth = 1;
        g.font = "12px ui-monospace, monospace";
        g.textAlign = "right";
        g.textBaseline = "middle";
        for (let tick = 0; tick <= 4; tick++) {
          const mag = low + ((high - low) * tick) / 4;
          const y = yAt(mag);
          g.beginPath();
          g.moveTo(pad.left, y);
          g.lineTo(width - pad.right, y);
          g.stroke();
          g.fillText(mag.toFixed(2), pad.left - 10, y);
        }

        g.save();
        g.translate(18, (pad.top + height - pad.bottom) / 2);
        g.rotate(-Math.PI / 2);
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillStyle = "#64748b";
        g.fillText("AB mag", 0, 0);
        g.restore();

        g.textBaseline = "top";
        g.textAlign = "left";
        const first = points[0].date ? points[0].date.slice(0, 10) : "";
        const last = points[points.length - 1].date ? points[points.length - 1].date.slice(0, 10) : "";
        g.fillText(first, pad.left, height - pad.bottom + 10);
        g.textAlign = "right";
        g.fillText(last, width - pad.right, height - pad.bottom + 10);

        g.strokeStyle = "rgba(94,234,212,0.55)";
        g.lineWidth = 2;
        g.beginPath();
        points.forEach((point, i) => {
          const x = xAt(i);
          const y = yAt(point.mag);
          if (i) g.lineTo(x, y);
          else g.moveTo(x, y);
        });
        g.stroke();

        const currentDate = S.frames[S.index]?.date;
        points.forEach((point, i) => {
          const x = xAt(i);
          const y = yAt(point.mag);
          const error = errors[i];
          if (error > 0) {
            g.strokeStyle = "rgba(251,191,36,0.55)";
            g.lineWidth = 1.5;
            g.beginPath();
            g.moveTo(x, yAt(point.mag - error));
            g.lineTo(x, yAt(point.mag + error));
            g.stroke();
          }
          const active = point.date && point.date === currentDate;
          g.fillStyle = active ? "#fbbf24" : "#5eead4";
          g.beginPath();
          g.arc(x, y, active ? 5 : 3.5, 0, Math.PI * 2);
          g.fill();
        });
      }

      // ------------------------------------------------------------ name search
      async function resolveObject() {
        const input = $("objectName");
        const name = input.value.trim();
        if (!name) {
          showError("error.name");
          return false;
        }
        const button = $("resolveBtn");
        button.disabled = true;
        button.textContent = t("status.resolving");
        clearError();
        setStatus("status.resolving");
        try {
          const data = await getJSON(`/api/resolve?name=${encodeURIComponent(name)}`, 20000);
          S.object = data.name || name;
          input.value = S.object;
          setCoords(Number(data.ra), Number(data.dec));
          showToast(
            t("search.resolved", {
              name: S.object,
              resolver: data.resolver || "Sesame",
            })
          );
          await search();
          return true;
        } catch (error) {
          const message = String(error?.message || error);
          const notFound = /no object was found|nothing found|404/i.test(message);
          showError(notFound ? "error.resolveNotFound" : "error.resolve", "", { name });
          setStatus("status.idle");
          return false;
        } finally {
          button.disabled = false;
          button.textContent = t("search.resolve");
        }
      }

      // ---------------------------------------------------------------- zoom
      const ZOOM_MIN = 1;
      const ZOOM_MAX = 8;

      function viewRect() {
        const z = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, S.zoom));
        const size = canvas.width / z;
        const maxOffset = canvas.width - size;
        const sx = Math.min(maxOffset, Math.max(0, S.cx * canvas.width - size / 2));
        const sy = Math.min(maxOffset, Math.max(0, S.cy * canvas.height - size / 2));
        return { sx, sy, size, z };
      }

      function normalizeView() {
        const view = viewRect();
        S.cx = (view.sx + view.size / 2) / canvas.width;
        S.cy = (view.sy + view.size / 2) / canvas.height;
      }

      function updateZoomUI() {
        const level = S.zoom >= 4 ? S.zoom.toFixed(0) : S.zoom.toFixed(1).replace(/\.0$/, "");
        $("zoomLevel").textContent = `${level}×`;
        $("zoomOut").disabled = S.zoom <= ZOOM_MIN;
        $("zoomIn").disabled = S.zoom >= ZOOM_MAX;
        $("zoomReset").disabled = S.zoom <= ZOOM_MIN;
        $("viewerWrap").classList.toggle("zoomed", S.zoom > ZOOM_MIN);
      }

      function updateCrosshair() {
        const wrap = $("crosshair");
        if (!S.cross || !S.frames.length || !S.images[S.index]) {
          wrap.classList.add("hidden");
          return;
        }
        const view = viewRect();
        const x = (canvas.width / 2 - view.sx) / view.size;
        const y = (canvas.height / 2 - view.sy) / view.size;
        const inside = x >= 0 && x <= 1 && y >= 0 && y <= 1;
        wrap.classList.toggle("hidden", !inside);
        if (!inside) return;
        const xp = `${(x * 100).toFixed(3)}%`;
        const yp = `${(y * 100).toFixed(3)}%`;
        $("crosshairV").style.left = xp;
        $("crosshairH").style.top = yp;
        $("crosshairDot").style.left = xp;
        $("crosshairDot").style.top = yp;
      }

      function setZoom(next, anchor = null) {
        const clamped = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
        if (anchor) {
          const view = viewRect();
          const focusX = view.sx + (anchor.x / canvas.width) * view.size;
          const focusY = view.sy + (anchor.y / canvas.height) * view.size;
          const size = canvas.width / clamped;
          S.cx = (focusX - (anchor.x / canvas.width) * size + size / 2) / canvas.width;
          S.cy = (focusY - (anchor.y / canvas.height) * size + size / 2) / canvas.height;
        }
        S.zoom = clamped;
        normalizeView();
        updateZoomUI();
        updateCrosshair();
        render();
        syncUrl();
      }

      function panBy(fx, fy) {
        if (S.zoom <= ZOOM_MIN) return;
        const view = viewRect();
        S.cx += (fx * view.size) / canvas.width;
        S.cy += (fy * view.size) / canvas.height;
        normalizeView();
        updateCrosshair();
        render();
      }

      function resetZoom() {
        S.zoom = ZOOM_MIN;
        S.cx = 0.5;
        S.cy = 0.5;
        updateZoomUI();
        updateCrosshair();
        render();
        syncUrl();
      }

      // ----------------------------------------------------------- observations
      function bestBand() {
        const ranked = [...S.bands].sort((a, b) => b.count - a.count || a.band - b.band);
        return ranked.length ? ranked[0].band : null;
      }

      // Retry IRSA 502/503 and transient network errors.
      const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
      const OBS_RETRY_DELAYS = [1500, 3000, 5000];

      async function fetchObservations(band, isStale = () => false) {
        const query = new URLSearchParams({ ra: String(S.ra), dec: String(S.dec), radius: "0.05" });
        if (band) query.set("band", String(band));
        const url = `/api/observations?${query.toString()}`;
        for (let attempt = 0; ; attempt++) {
          try {
            return await getJSON(url);
          } catch (error) {
            const retryable =
              !error.isAbort && (error.isNetwork || RETRYABLE_STATUS.has(error.status));
            if (!retryable || attempt >= OBS_RETRY_DELAYS.length || isStale()) throw error;
            setStatus("status.retrying", { n: attempt + 1, max: OBS_RETRY_DELAYS.length });
            await new Promise((resolve) => setTimeout(resolve, OBS_RETRY_DELAYS[attempt]));
            if (isStale()) throw error;
          }
        }
      }

      async function search() {
        const ra = Number($("ra").value);
        const dec = Number($("dec").value);
        if (!Number.isFinite(ra) || ra < 0 || ra >= 360) {
          showError("error.ra");
          return;
        }
        if (!Number.isFinite(dec) || dec < -90 || dec > 90) {
          showError("error.dec");
          return;
        }

        S.ra = ra;
        S.dec = dec;
        S.fov = Number($("fov").value);
        S.cmap = $("cmap").value;
        syncUrl();
        updateLinks();
        clearError();
        stopPlayback();
        stopBlink();
        S.allFrames = [];
        S.frames = [];
        S.images = [];
        S.bands = [];
        S.autoBand = null;
        S.diffCache.clear();
        S.science = null;
        S.photo = null;
        S.scienceID++;
        S.filterID++;
        S.loadID++;
        S.index = 0;
        setBusy(true);
        setStatus("status.searching");
        showLoader(true, 0, 0);
        $("emptyState").classList.add("hidden");
        $("emptyState").classList.remove("grid");
        drawPlaceholder();
        updateCrosshair();

        const requestID = ++S.requestID;
        try {
          const data = await fetchObservations(null, () => requestID !== S.requestID);
          if (requestID !== S.requestID) return;

          S.allFrames = data.frames || [];
          S.bands = data.bands || [];
          S.total = data.total ?? S.allFrames.length;
          S.truncated = !!data.truncated;
          S.allTotal = S.total;
          S.allTruncated = S.truncated;
          rebuildBandOptions();

          if (!S.allFrames.length) {
            showLoader(false);
            showEmpty("viewer.emptyTitle", "viewer.emptyBody");
            setStatus("status.empty", null, "error");
            updateControls();
            return;
          }

          await applyFilters();
        } catch (error) {
          if (requestID !== S.requestID) return;
          showLoader(false);
          const raw = String(error?.message || error);
          if (error?.isNetwork) {
            showError("error.network", raw);
          } else if (error?.isAbort) {
            showError("error.observations");
          } else {
            showError("error.upstream", String(error?.detail || raw));
          }
          showEmpty("viewer.errorTitle", "viewer.errorBody");
          setStatus("status.idle");
          updateControls();
        } finally {
          if (requestID === S.requestID) setBusy(false);
        }
      }

      async function applyFilters() {
        S.fov = Number($("fov").value);
        S.cmap = $("cmap").value;
        syncUrl();
        updateLinks();
        const mode = $("band").value;
        const band = mode === "all" ? null : mode === "auto" ? bestBand() : Number(mode);
        S.autoBand = mode === "auto" ? band : null;
        $("autoBandNote").textContent = S.autoBand ? t("status.autoBand", { band }) : "";
        clearError();

        const requestID = S.requestID;
        const filterID = ++S.filterID;
        let frames = [];
        let total = 0;
        let truncated = false;
        if (band === null) {
          frames = S.allFrames.slice();
          total = S.allTotal;
          truncated = S.allTruncated;
        } else {
          try {
            const data = await fetchObservations(
              band,
              () => requestID !== S.requestID || filterID !== S.filterID
            );
            if (requestID !== S.requestID || filterID !== S.filterID) return;
            frames = data.frames || [];
            total = data.total ?? frames.length;
            truncated = !!data.truncated;
            S.total = total;
            S.truncated = truncated;
          } catch (error) {
            if (requestID !== S.requestID || filterID !== S.filterID) return;
            showLoader(false);
            const raw = String(error?.message || error);
            if (error?.isNetwork) {
              showError("error.network", raw);
            } else if (error?.isAbort) {
              showError("error.observations");
            } else {
              showError("error.upstream", String(error?.detail || raw));
            }
            showEmpty("viewer.errorTitle", "viewer.errorBody");
            setStatus("status.idle");
            updateControls();
            return;
          }
        }

        S.frames = frames;
        if (!frames.length) {
          stopPlayback();
          stopBlink();
          S.images = [];
          S.index = 0;
          updateControls();
          updateHud();
          showEmpty("viewer.emptyTitle", "viewer.emptyBody");
          setStatus("status.emptyBand", null, "error");
          return;
        }
        const statusKey =
          band === null
            ? truncated
              ? "status.sampled"
              : "status.found"
            : truncated
            ? "status.sampledBand"
            : "status.bandFrames";
        const statusVars =
          band === null
            ? { n: frames.length, total, bands: S.bands.length }
            : { n: frames.length, total, band };
        setStatus(statusKey, statusVars, "ok");
        loadFrames();
      }

      // ------------------------------------------------------------- frame load
      function loadImage(src) {
        return new Promise((resolve, reject) => {
          const image = new Image();
          image.decoding = "async";
          image.onload = () => resolve(image);
          image.onerror = () => reject(new Error("image"));
          image.src = src;
        });
      }

      async function loadFrames() {
        stopPlayback();
        stopBlink();
        const loadID = ++S.loadID;
        S.images = new Array(S.frames.length).fill(null);
        S.index = 0;
        S.loaded = 0;
        S.failed = 0;
        S.started = false;
        S.loading = true;
        S.diffCache.clear();
        if (S.mode === "science") {
          S.scienceID++;
          S.science = null;
          S.photo = null;
          renderScience();
        }
        $("emptyState").classList.add("hidden");
        $("emptyState").classList.remove("grid");
        showLoader(true, 0, S.frames.length);
        updateControls();
        updateHud();
        drawPlaceholder();

        const fov = S.fov;
        const cmap = S.cmap;
        const total = S.frames.length;
        const restoreIndex = S.pendingFrame;
        let next = 0;

        const worker = async () => {
          while (next < total) {
            const i = next++;
            let image = null;
            try {
              image = await loadImage(frameUrl(S.frames[i], fov, cmap));
            } catch {
              if (loadID === S.loadID) S.failed++;
            }
            if (loadID !== S.loadID) return;
            S.images[i] = image;
            S.loaded++;
            const done = S.images.filter(Boolean).length;
            updateProgress(done, total);
            if (!S.started && done >= 2) {
              S.started = true;
              const firstReady = S.images.findIndex(Boolean);
              S.index = Math.max(0, firstReady);
              showLoader(false);
              render();
              if (restoreIndex == null) setPlaying(true);
            } else if (!S.started && done === 1) {
              S.index = S.images.findIndex(Boolean);
              render();
            }
          }
        };

        await Promise.all(Array.from({ length: Math.min(6, total) }, worker));
        if (loadID !== S.loadID) return;

        // IRSA throttles bursts; retry missing frames in small batches.
        for (let pass = 0; S.failed > 0 && pass < 2 && loadID === S.loadID; pass++) {
          await new Promise((resolve) => setTimeout(resolve, 1200 * (pass + 1)));
          if (loadID !== S.loadID) return;
          const missing = S.images.map((image, i) => (image ? -1 : i)).filter((i) => i >= 0);
          let cursor = 0;
          const retryWorker = async () => {
            while (cursor < missing.length) {
              const i = missing[cursor++];
              let image = null;
              try {
                image = await loadImage(frameUrl(S.frames[i], fov, cmap));
              } catch {}
              if (loadID !== S.loadID) return;
              if (image) {
                S.images[i] = image;
                S.failed = Math.max(0, S.failed - 1);
                if (i === S.index) render();
              }
              updateProgress(S.images.filter(Boolean).length, total);
            }
          };
          await Promise.all(Array.from({ length: Math.min(3, missing.length) }, retryWorker));
        }
        if (loadID !== S.loadID) return;

        S.loading = false;
        showLoader(false);
        if (!S.started) {
          const firstReady = S.images.findIndex(Boolean);
          if (firstReady >= 0) {
            S.index = firstReady;
            render();
          } else {
            showEmpty("viewer.errorTitle", "viewer.errorBody");
            setStatus("status.imageErrors", { n: S.failed }, "error");
          }
        }
        if (restoreIndex != null) {
          const wanted = Math.min(restoreIndex, S.images.length - 1);
          const target = S.images[wanted] ? wanted : S.images.findIndex(Boolean);
          if (target >= 0) {
            stopPlayback();
            S.index = target;
            render();
          }
        }
        S.pendingFrame = null;
        if (S.failed) setStatus("status.imageErrors", { n: S.failed }, "muted");
        updateControls();
        updateHud();
        if (S.mode === "science") loadScience();
      }

      function updateProgress(done, total) {
        const percent = total ? Math.round((done / total) * 100) : 0;
        $("loaderBar").style.width = `${percent}%`;
        $("loadBar").style.width = `${percent}%`;
        $("loaderText").textContent = t("loader.progress", { done, total });
      }

      function showLoader(show, done = 0, total = 0) {
        $("loader").classList.toggle("hidden", !show);
        if (show) updateProgress(done, total);
        else $("loader").classList.add("hidden");
      }

      function showEmpty(titleKey, bodyKey) {
        $("emptyTitle").dataset.i18n = titleKey;
        $("emptyBody").dataset.i18n = bodyKey;
        $("emptyTitle").textContent = t(titleKey);
        $("emptyBody").textContent = t(bodyKey);
        $("emptyState").classList.remove("hidden");
        $("emptyState").classList.add("grid");
      }

      function drawPlaceholder() {
        ctx.fillStyle = "#05090f";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "rgba(94,234,212,0.16)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 8; i++) {
          const p = (i / 8) * canvas.width;
          ctx.beginPath();
          ctx.moveTo(p, 0);
          ctx.lineTo(p, canvas.height);
          ctx.moveTo(0, p);
          ctx.lineTo(canvas.width, p);
          ctx.stroke();
        }
      }

      // --------------------------------------------------------------- playback
      function setPlaying(playing) {
        S.playing = playing;
        clearInterval(S.playTimer);
        S.playTimer = null;
        if (playing) {
          const interval = Math.max(120, 900 / S.speed);
          S.playTimer = setInterval(() => step(1), interval);
        }
        updatePlayButton(!playing);
      }

      function updatePlayButton(showPlay) {
        const btn = $("playBtn");
        btn.textContent = showPlay ? t("ctrl.play") : t("ctrl.pause");
      }

      function stopPlayback() {
        S.playing = false;
        clearInterval(S.playTimer);
        S.playTimer = null;
        updatePlayButton(true);
      }

      function step(direction) {
        if (!S.frames.length) return;
        const total = S.frames.length;
        for (let attempt = 1; attempt <= total; attempt++) {
          const target = (S.index + direction * attempt + total * 2) % total;
          if (S.images[target]) {
            S.index = target;
            render();
            return;
          }
        }
      }

      function startBlink() {
        if (S.frames.length < 2) {
          showToast(t("blink.needTwo"));
          return;
        }
        stopPlayback();
        clearInterval(S.blinkTimer);
        S.blinking = true;
        $("blinkBtn").classList.add("active");
        $("blinkBtn").textContent = t("ctrl.blinkOn");
        let last = false;
        S.index = 0;
        render();
        S.blinkTimer = setInterval(() => {
          last = !last;
          const candidates = last ? [...S.images.keys()].reverse() : [...S.images.keys()];
          const target = candidates.find((i) => S.images[i]);
          if (target !== undefined) {
            S.index = target;
            render();
          }
        }, 400);
      }

      function stopBlink() {
        clearInterval(S.blinkTimer);
        S.blinkTimer = null;
        S.blinking = false;
        $("blinkBtn").classList.remove("active");
        $("blinkBtn").textContent = t("ctrl.blink");
      }

      // --------------------------------------------------------------- rendering
      function luminance(data, index) {
        return 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
      }

      function diffThreshold(deltas) {
        const sample = [];
        for (let i = 0; i < deltas.length; i += 4) sample.push(Math.abs(deltas[i]));
        sample.sort((a, b) => a - b);
        const median = sample[Math.floor(sample.length * 0.5)] || 0;
        const deviations = sample.map((value) => Math.abs(value - median)).sort((a, b) => a - b);
        const mad = deviations[Math.floor(deviations.length * 0.5)] || 0;
        return Math.max(12, Math.min(48, median + 8 * 1.4826 * mad));
      }

      function findDiffBlobs(deltas, size) {
        const n = size * size;
        const threshold = diffThreshold(deltas);
        const visited = new Uint8Array(n);
        const stack = new Int32Array(n);
        const raw = [];
        for (let start = 0; start < n; start++) {
          if (visited[start] || Math.abs(deltas[start]) < threshold) continue;
          let top = 0;
          stack[top++] = start;
          visited[start] = 1;
          let area = 0;
          let sum = 0;
          let sx = 0;
          let sy = 0;
          let peak = 0;
          let minX = size;
          let maxX = -1;
          let minY = size;
          let maxY = -1;
          while (top > 0) {
            const p = stack[--top];
            const x = p % size;
            const y = (p - x) / size;
            const weight = Math.abs(deltas[p]);
            area += 1;
            sum += weight;
            sx += x * weight;
            sy += y * weight;
            if (weight > peak) peak = weight;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            for (let dy = -1; dy <= 1; dy++) {
              const ny = y + dy;
              if (ny < 0 || ny >= size) continue;
              for (let dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                const nx = x + dx;
                if (nx < 0 || nx >= size) continue;
                const q = ny * size + nx;
                if (visited[q] || Math.abs(deltas[q]) < threshold) continue;
                visited[q] = 1;
                stack[top++] = q;
              }
            }
          }
          if (area >= 4 && area <= n * 0.08 && sum > 0) {
            raw.push({
              x: sx / sum,
              y: sy / sum,
              area,
              sum,
              peak,
              minX,
              maxX,
              minY,
              maxY,
              r: Math.max(7, Math.hypot(maxX - minX + 1, maxY - minY + 1) / 2 + 4),
            });
          }
        }

        // Merge components belonging to the same source/PSF before drawing.
        raw.sort((a, b) => b.sum - a.sum);
        const merged = [];
        for (const blob of raw.slice(0, 80)) {
          const hit = merged.find(
            (item) => Math.hypot(item.x - blob.x, item.y - blob.y) <= item.r + blob.r + 5
          );
          if (!hit) {
            merged.push({ ...blob });
            continue;
          }
          const total = hit.sum + blob.sum;
          hit.x = (hit.x * hit.sum + blob.x * blob.sum) / total;
          hit.y = (hit.y * hit.sum + blob.y * blob.sum) / total;
          hit.sum = total;
          hit.area += blob.area;
          hit.peak = Math.max(hit.peak, blob.peak);
          hit.minX = Math.min(hit.minX, blob.minX);
          hit.maxX = Math.max(hit.maxX, blob.maxX);
          hit.minY = Math.min(hit.minY, blob.minY);
          hit.maxY = Math.max(hit.maxY, blob.maxY);
          hit.r = Math.max(
            hit.r,
            Math.hypot(hit.maxX - hit.minX + 1, hit.maxY - hit.minY + 1) / 2 + 4
          );
        }
        return merged
          .filter((item) => item.area >= 5 && item.r <= 140)
          .sort((a, b) => b.sum - a.sum)
          .slice(0, 8)
          .map((item) => ({ x: item.x, y: item.y, r: Math.min(item.r, 120) }));
      }

      function drawDiffCircles(blobs) {
        if (!blobs.length) return;
        const view = viewRect();
        const toCanvas = canvas.width / 400;
        const zoom = canvas.width / view.size;
        ctx.save();
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "rgba(251,191,36,0.95)";
        ctx.fillStyle = "rgba(251,191,36,0.95)";
        ctx.shadowColor = "rgba(251,191,36,0.7)";
        ctx.shadowBlur = 9;
        for (const blob of blobs) {
          const bx = blob.x * toCanvas;
          const by = blob.y * toCanvas;
          const x = (bx - view.sx) * zoom;
          const y = (by - view.sy) * zoom;
          const radius = Math.max(8, blob.r * toCanvas * zoom);
          if (x + radius < 0 || y + radius < 0 || x - radius > canvas.width || y - radius > canvas.height) {
            continue;
          }
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      function drawCropped(source) {
        const view = viewRect();
        const k = source.width / canvas.width;
        ctx.drawImage(
          source,
          view.sx * k,
          view.sy * k,
          view.size * k,
          view.size * k,
          0,
          0,
          canvas.width,
          canvas.height
        );
      }

      function buildDiff(current, base) {
        const key = `${S.cmap}|${S.index}|${S.fov}|${S.frames[S.index]?.id}|${base.src}`;
        const cached = S.diffCache.get(key);
        if (cached) return cached;

        const size = 400;
        const canA = document.createElement("canvas");
        const canB = document.createElement("canvas");
        canA.width = canB.width = size;
        canA.height = canB.height = size;
        const cA = canA.getContext("2d", { willReadFrequently: true });
        const cB = canB.getContext("2d", { willReadFrequently: true });
        cA.drawImage(base, 0, 0, size, size);
        cB.drawImage(current, 0, 0, size, size);
        const a = cA.getImageData(0, 0, size, size).data;
        const b = cB.getImageData(0, 0, size, size).data;
        const deltas = new Float32Array(size * size);
        const hist = new Uint32Array(256);
        for (let i = 0, p = 0; i < size * size; i++, p += 4) {
          const delta = luminance(b, p) - luminance(a, p);
          deltas[i] = delta;
          hist[Math.min(255, Math.round(Math.abs(delta)))]++;
        }
        const target = Math.round(size * size * 0.995);
        let acc = 0;
        let scale = 1;
        for (let i = 0; i < 256; i++) {
          acc += hist[i];
          if (acc >= target) {
            scale = Math.max(1, i);
            break;
          }
        }
        const out = document.createElement("canvas");
        out.width = out.height = size;
        const outCtx = out.getContext("2d");
        const image = outCtx.createImageData(size, size);
        const d = image.data;
        for (let i = 0, p = 0; i < size * size; i++, p += 4) {
          const value = Math.max(0, Math.min(255, Math.round(128 + (deltas[i] / scale) * 127)));
          d[p] = d[p + 1] = d[p + 2] = value;
          d[p + 3] = 255;
        }
        outCtx.putImageData(image, 0, 0);
        const result = { canvas: out, circles: findDiffBlobs(deltas, size) };
        S.diffCache.set(key, result);
        return result;
      }

      function render() {
        if (!S.frames.length) {
          drawPlaceholder();
          $("diffCount").textContent = "";
          $("diffWarn").classList.add("hidden");
          updateCrosshair();
          return;
        }
        const current = S.images[S.index];
        if (!current) {
          drawPlaceholder();
          $("diffWarn").classList.add("hidden");
          updateHud();
          updateCrosshair();
          return;
        }
        const baseIndex = S.images.findIndex(Boolean);
        const base = baseIndex >= 0 ? S.images[baseIndex] : null;
        const showDiff = S.diff && base && S.index !== baseIndex;
        let circles = [];
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (showDiff) {
          const diff = buildDiff(current, base);
          drawCropped(diff.canvas);
          if (S.circles) circles = diff.circles;
        } else {
          drawCropped(current);
        }
        drawDiffCircles(circles);
        $("diffBadge").classList.toggle("hidden", !showDiff);
        $("diffWarn").classList.toggle("hidden", !(S.diff && $("band").value === "all"));
        $("diffCount").textContent = showDiff
          ? S.circles
            ? circles.length
              ? t("hud.regions", { n: circles.length })
              : t("hud.noRegions")
            : ""
          : "";
        updateHud();
        updateFrameInfo();
        updateCrosshair();
        $("timeline").value = String(S.index);
        if (S.mode === "science" && S.photo) drawCurve();
        scheduleSyncUrl();
      }

      function updateHud() {
        const frame = S.frames[S.index];
        if (!frame) {
          $("hudMission").textContent = "SPHEREx —";
          $("hudDate").textContent = "—";
          $("hudWave").textContent = "— µm";
          $("hudFov").textContent = "—";
          $("hudCoords").textContent = "—";
          $("hudFrame").textContent = "—";
          return;
        }
        const survey = frame.collection === "spherex_dr1" ? "DR1" : "QR3";
        $("hudMission").textContent = `SPHEREx ${survey}`;
        $("hudDate").textContent = fmtDate(frame.date);
        $("hudWave").textContent = frame.wavelength_um ? `${frame.wavelength_um.toFixed(2)} µm` : "— µm";
        $("hudFov").textContent = `${(S.fov * 60).toFixed(1)}′ × ${(S.fov * 60).toFixed(1)}′`;
        $("hudCoords").textContent = `${S.ra.toFixed(3)}°, ${S.dec >= 0 ? "+" : ""}${S.dec.toFixed(3)}°`;
        $("hudFrame").textContent = `${S.index + 1} / ${S.frames.length}`;
      }

      function updateFrameInfo() {
        const frame = S.frames[S.index];
        $("frameInfo").textContent = frame
          ? t("ctrl.frameInfo", {
              i: S.index + 1,
              n: S.frames.length,
              date: frame.date ? frame.date.slice(0, 10) : "—",
            })
          : "—";
      }

      function updateControlsLabels() {
        updatePlayButton(!S.playing);
        $("prevBtn").textContent = t("ctrl.prev");
        $("nextBtn").textContent = t("ctrl.next");
        $("blinkBtn").textContent = S.blinking ? t("ctrl.blinkOn") : t("ctrl.blink");
        if (S.autoBand) $("autoBandNote").textContent = t("status.autoBand", { band: S.autoBand });
      }

      function updateControls() {
        const has = S.frames.length > 0;
        $("playBtn").disabled = !has;
        $("prevBtn").disabled = !has;
        $("nextBtn").disabled = !has;
        $("timeline").disabled = !has;
        $("gifBtn").disabled = !has;
        $("timeline").max = String(Math.max(0, S.frames.length - 1));
        $("timeline").value = String(Math.min(S.index, Math.max(0, S.frames.length - 1)));
        updateControlsLabels();
      }

      // --------------------------------------------------------------- events
      $("searchForm").addEventListener("submit", (event) => {
        event.preventDefault();
        const name = $("objectName").value.trim();
        if (name && name !== S.object) resolveObject();
        else search();
      });
      $("resolveBtn").addEventListener("click", () => resolveObject());
      $("objectName").addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        resolveObject();
      });
      ["ra", "dec"].forEach((id) => {
        $(id).addEventListener("input", () => {
          if (!S.object) return;
          S.object = "";
          $("objectName").value = "";
        });
      });
      $("retryBtn").addEventListener("click", () => search());
      $("playBtn").addEventListener("click", () => {
        if (!S.frames.length) return;
        if (S.blinking) stopBlink();
        if (S.playing) stopPlayback();
        else {
          setPlaying(true);
          step(1);
        }
      });
      $("prevBtn").addEventListener("click", () => {
        stopPlayback();
        step(-1);
      });
      $("nextBtn").addEventListener("click", () => {
        stopPlayback();
        step(1);
      });
      $("speed").addEventListener("change", (event) => {
        S.speed = Number(event.target.value);
        syncUrl();
        if (S.playing) setPlaying(true);
      });
      $("blinkBtn").addEventListener("click", () => {
        if (S.blinking) {
          stopBlink();
          render();
        } else startBlink();
      });
      $("diffToggle").addEventListener("change", (event) => {
        S.diff = event.target.checked;
        S.diffCache.clear();
        render();
        syncUrl();
      });
      $("circlesToggle").addEventListener("change", (event) => {
        S.circles = event.target.checked;
        render();
        syncUrl();
      });
      $("crosshairToggle").addEventListener("change", (event) => {
        S.cross = event.target.checked;
        updateCrosshair();
        syncUrl();
      });
      $("timeline").addEventListener("input", (event) => {
        stopPlayback();
        S.index = Number(event.target.value);
        render();
        syncUrl();
      });
      $("fov").addEventListener("change", () => {
        if (S.frames.length || S.allFrames.length) applyFilters();
        else {
          S.fov = Number($("fov").value);
          syncUrl();
          updateLinks();
          updateHud();
        }
      });
      $("cmap").addEventListener("change", () => {
        S.cmap = $("cmap").value;
        if (S.frames.length) applyFilters();
        else {
          syncUrl();
          updateLooks();
        }
      });
      $("band").addEventListener("change", () => {
        if (!S.allFrames.length) return;
        applyFilters();
      });

      // ------------------------------------------------------------------ zoom
      $("zoomIn").addEventListener("click", () => setZoom(S.zoom * 1.6));
      $("zoomOut").addEventListener("click", () => setZoom(S.zoom / 1.6));
      $("zoomReset").addEventListener("click", resetZoom);

      $("viewerWrap").addEventListener(
        "wheel",
        (event) => {
          if (!S.frames.length) return;
          event.preventDefault();
          const rect = canvas.getBoundingClientRect();
          const anchor = {
            x: ((event.clientX - rect.left) / rect.width) * canvas.width,
            y: ((event.clientY - rect.top) / rect.height) * canvas.height,
          };
          setZoom(S.zoom * Math.exp(-event.deltaY * 0.0015), anchor);
        },
        { passive: false }
      );

      let drag = null;
      $("viewerWrap").addEventListener("pointerdown", (event) => {
        if (!S.frames.length || S.zoom <= ZOOM_MIN || event.button !== 0) return;
        drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
        $("viewerWrap").classList.add("dragging");
        try {
          $("viewerWrap").setPointerCapture(event.pointerId);
        } catch {}
      });
      $("viewerWrap").addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.id) return;
        const rect = canvas.getBoundingClientRect();
        const fx = (event.clientX - drag.x) / rect.width;
        const fy = (event.clientY - drag.y) / rect.height;
        drag.x = event.clientX;
        drag.y = event.clientY;
        panBy(-fx, -fy);
      });
      const endDrag = (event) => {
        if (!drag || (event && event.pointerId !== drag.id)) return;
        drag = null;
        $("viewerWrap").classList.remove("dragging");
        syncUrl();
      };
      $("viewerWrap").addEventListener("pointerup", endDrag);
      $("viewerWrap").addEventListener("pointercancel", endDrag);
      $("viewerWrap").addEventListener("dblclick", () => {
        if (S.zoom > ZOOM_MIN) resetZoom();
      });
      document.querySelectorAll("#modeSwitch button").forEach((button) => {
        button.addEventListener("click", () => setMode(button.dataset.mode));
      });
      document.querySelectorAll("#lang button").forEach((button) => {
        button.addEventListener("click", () => setLang(button.dataset.lang));
      });
      $("shareBtn").addEventListener("click", async () => {
        syncUrl();
        try {
          await navigator.clipboard.writeText(location.href);
          showToast(t("actions.copied"));
        } catch {
          showToast(t("actions.copyfail", { url: location.href }));
        }
      });
      $("gifBtn").addEventListener("click", exportGif);

      async function exportGif() {
        if (S.frames.length < 2) {
          showToast(t("actions.gifFew"));
          return;
        }
        const button = $("gifBtn");
        const original = t("actions.gif");
        button.disabled = true;
        button.textContent = t("actions.gifBusy");
        try {
          const selected = S.frames.slice(0, 24);
          const response = await fetch("/api/gif", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              fits_urls: selected.map((frame) => frame.access_url),
              ra: S.ra,
              dec: S.dec,
              size: S.fov,
              colormap: S.cmap,
              delay_ms: 400,
            }),
          });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = `spherex-${S.ra.toFixed(2)}_${S.dec.toFixed(2)}.gif`;
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
          showToast(t("actions.gifDone"));
        } catch {
          showToast(t("actions.gifFail"));
        } finally {
          button.disabled = false;
          button.textContent = original;
        }
      }

      function updateLooks() {
        updateCrosshair();
      }

      // ----------------------------------------------------------------- boot
      function readUrlState() {
        const params = new URLSearchParams(location.search);
        const lang = params.get("lang");
        if (lang === "es" || lang === "en") {
          LANG = lang;
          try {
            localStorage.setItem("pxs-lang", LANG);
          } catch {}
        }

        let savedMode = null;
        try {
          savedMode = localStorage.getItem("pxs-mode");
        } catch {}
        const mode = params.get("mode");
        if (mode === "science" || mode === "casual") S.mode = mode;
        else if (savedMode === "science" || savedMode === "casual") S.mode = savedMode;

        const fov = params.get("fov");
        if (["0.02", "0.05", "0.1"].includes(fov)) $("fov").value = fov;
        const cmap = params.get("cmap");
        if (["gray", "inferno", "viridis"].includes(cmap)) $("cmap").value = cmap;
        const band = params.get("band");

        S.diff = params.get("diff") === "1";
        S.circles = params.get("circles") !== "0";
        S.cross = params.get("cross") !== "0";
        $("diffToggle").checked = S.diff;
        $("circlesToggle").checked = S.circles;
        $("crosshairToggle").checked = S.cross;

        const speed = Number(params.get("speed"));
        if ([1, 2, 5].includes(speed)) {
          S.speed = speed;
          $("speed").value = String(speed);
        }

        const zoom = Number(params.get("z"));
        if (Number.isFinite(zoom) && zoom > 0) {
          S.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom));
        }
        const cx = Number(params.get("cx"));
        const cy = Number(params.get("cy"));
        if (params.has("cx") && params.has("cy") && Number.isFinite(cx) && Number.isFinite(cy)) {
          S.cx = Math.min(1, Math.max(0, cx));
          S.cy = Math.min(1, Math.max(0, cy));
        }

        const frame = Number(params.get("frame"));
        if (params.has("frame") && Number.isInteger(frame) && frame >= 0) S.pendingFrame = frame;

        const object = (params.get("object") || "").trim();
        if (object) {
          S.object = object;
          $("objectName").value = object;
        }

        const ra = Number(params.get("ra"));
        const dec = Number(params.get("dec"));
        const valid =
          params.has("ra") &&
          params.has("dec") &&
          Number.isFinite(ra) &&
          Number.isFinite(dec) &&
          ra >= 0 &&
          ra < 360 &&
          dec >= -90 &&
          dec <= 90;
        setCoords(valid ? ra : 269.452, valid ? dec : 4.693);

        S.fov = Number($("fov").value);
        S.cmap = $("cmap").value;
        normalizeView();
        return { band };
      }

      function boot() {
        const state = readUrlState();
        renderPresets();
        setLang(LANG);
        setMode(S.mode, false);
        if (state.band) S.pendingBand = state.band;
        updateZoomUI();
        updateCrosshair();
        updateLinks();
        updateControls();
        drawPlaceholder();
        search();
      }

      boot();
    </script>
  </body>
</html>
'''
