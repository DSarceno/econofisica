# --- FILE: src/data/loaders.py ---
"""
Adquisicion de series de precios para el activo de estudio.

Trazabilidad tesis -> codigo:
    - Justificacion del activo: tch3 seccion "Seleccion del indice S&P 500 y SPY".
    - Variable de interes: precio adjustado P_t (luego transformado a log-retornos
      en src/preprocessing/returns.py, conforme a tch3 ec. log-return).

Estrategia:
    1. Cache local en data/raw/<ticker>_<start>_<end>.parquet.
    2. Si cache existe -> cargar; si no -> descargar via yfinance.
    3. Fallback automatico al ticker secundario si el primario falla.
    4. Hash SHA-256 del contenido para reproducibilidad.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore

from src.utils.io import ensure_dir, hash_array

logger = logging.getLogger(__name__)


class DataLoadError(RuntimeError):
    """Falla irrecuperable en adquisicion de datos."""


def _cache_path(cache_dir: str | Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "")
    return Path(cache_dir) / f"{safe}_{start}_{end}.parquet"


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise DataLoadError("yfinance no esta instalado")
    logger.info("Descargando %s desde yfinance [%s, %s]", ticker, start, end)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if df is None or df.empty:
        raise DataLoadError(f"yfinance retorno vacio para {ticker}")
    # Normalizar MultiIndex de columnas que aparece en versiones recientes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw",
    *,
    fallback_ticker: str | None = None,
    field: str = "Adj Close",
) -> pd.Series:
    """Devuelve la serie de precios (campo seleccionado) con cache local.

    Parameters
    ----------
    ticker, start, end :
        Identificador del activo y rango temporal (formato 'YYYY-MM-DD').
    cache_dir :
        Carpeta de cache local. Se crea si no existe.
    fallback_ticker :
        Si la descarga del primario falla, intenta este alternativo.
    field :
        Columna a extraer ('Adj Close' por defecto, alineado con literatura).
    """
    ensure_dir(cache_dir)
    cache = _cache_path(cache_dir, ticker, start, end)

    if cache.exists():
        logger.info("Cargando cache: %s", cache)
        df = pd.read_parquet(cache)
    else:
        try:
            df = _download_yfinance(ticker, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallo descarga primaria (%s): %s", ticker, exc)
            if fallback_ticker is None:
                raise DataLoadError(f"Imposible descargar {ticker}") from exc
            logger.info("Probando fallback: %s", fallback_ticker)
            df = _download_yfinance(fallback_ticker, start, end)
            cache = _cache_path(cache_dir, fallback_ticker, start, end)
        df.to_parquet(cache)
        logger.info("Cache escrito: %s", cache)

    if field not in df.columns:
        raise DataLoadError(f"Campo '{field}' ausente; disponibles: {list(df.columns)}")

    series = df[field].astype("float64").rename("price")
    digest = hash_array(series)
    logger.info("Datos cargados: n=%d, hash=%s", len(series), digest[:12])
    return series
