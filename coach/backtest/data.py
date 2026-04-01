from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from coach.data.adapters.local_csv import LocalCSVAdapter
from scripts.build_real_data import build_data, validate

_COACHAI_ARCHIVE_URLS = (
    "https://codeload.github.com/wywyWang/CoachAI-Projects/zip/refs/heads/main",
    "https://codeload.github.com/wywyWang/CoachAI-Projects/zip/refs/heads/master",
)


@dataclass(frozen=True)
class HistoricalDatasetPaths:
    players_csv: Path
    matches_csv: Path
    raw_set_dir: Path


@dataclass(frozen=True)
class MatchCutoff:
    cutoff_key: str
    cutoff_timestamp: pd.Timestamp
    policy: str


class ShuttleSetScraper:
    """Download the ShuttleSet archive from GitHub using standard-library requests."""

    def __init__(self, archive_urls: tuple[str, ...] = _COACHAI_ARCHIVE_URLS) -> None:
        self.archive_urls = archive_urls

    def download_archive(self, destination: str | Path) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)

        last_error: Exception | None = None
        for url in self.archive_urls:
            try:
                with urlopen(url, timeout=60) as response, dest.open("wb") as fh:
                    shutil.copyfileobj(response, fh)
                return dest
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                continue

        raise RuntimeError(
            "Unable to download ShuttleSet historical data from CoachAI-Projects."
        ) from last_error

    def extract_set_dir(self, archive_path: str | Path, destination: str | Path) -> Path:
        archive = Path(archive_path)
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive) as zf:
            root_member = next(
                (member.filename.split("/", 1)[0] for member in zf.infolist() if member.filename),
                None,
            )
            if root_member is None:
                raise RuntimeError(f"Archive {archive} is empty.")

            _safe_extract_zip(zf, dest)

        set_dir = dest / root_member / "ShuttleSet" / "set"
        if not set_dir.exists():
            raise RuntimeError(f"Could not locate ShuttleSet set directory under {dest}.")
        return set_dir

    def fetch_set_dir(self, destination: str | Path) -> Path:
        dest = Path(destination)
        archive_path = dest / "coachai_shuttleset.zip"
        archive = self.download_archive(archive_path)
        return self.extract_set_dir(archive, dest)


def _safe_extract_zip(zf: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in zf.infolist():
        member_path = destination / member.filename
        member_resolved = member_path.resolve()
        if not str(member_resolved).startswith(str(destination_resolved)):
            raise RuntimeError(f"Unsafe zip member path: {member.filename}")
    zf.extractall(destination)


def build_historical_dataset(
    *,
    raw_set_dir: str | Path,
    players_out: str | Path,
    matches_out: str | Path,
) -> HistoricalDatasetPaths:
    set_dir = Path(raw_set_dir)
    players_df, matches_df = build_data(set_dir)
    validate(players_df, matches_df)

    players_path = Path(players_out)
    matches_path = Path(matches_out)
    players_path.parent.mkdir(parents=True, exist_ok=True)
    matches_path.parent.mkdir(parents=True, exist_ok=True)

    players_df.to_csv(players_path, index=False)
    matches_df.to_csv(matches_path, index=False)
    return HistoricalDatasetPaths(players_csv=players_path, matches_csv=matches_path, raw_set_dir=set_dir)


def load_backtest_frames(
    players_path: str | Path,
    matches_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    players_df = pd.read_csv(players_path)
    matches_df = pd.read_csv(matches_path)
    matches_df["date"] = pd.to_datetime(matches_df["date"], utc=False)
    return players_df, matches_df


def prepare_chronological_matches(
    matches_df: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df["match_id"] = df.index.astype(str)
    df["actual_a_win"] = (df["winner_id"] == df["playerA_id"]).astype(int)

    timestamp_available = timestamp_column is not None and timestamp_column in df.columns
    if timestamp_available:
        df["sort_timestamp"] = pd.to_datetime(df[timestamp_column], utc=False)
        df["cutoff_timestamp"] = df["sort_timestamp"] - pd.Timedelta(seconds=1)
        df["cutoff_key"] = df["cutoff_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df["snapshot_policy"] = "exact_timestamp"
    else:
        df["sort_timestamp"] = df["date"]
        df["cutoff_timestamp"] = df["date"] - pd.Timedelta(days=1)
        df["cutoff_key"] = df["cutoff_timestamp"].dt.strftime("%Y-%m-%d")
        df["snapshot_policy"] = "previous_day_close"

    df["round_bucket"] = df["round"].map(normalize_round_bucket) if "round" in df.columns else "unknown"
    df["calendar_year"] = df["date"].dt.year.astype(int)
    sort_cols = ["sort_timestamp", "match_id"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def normalize_round_bucket(raw_round: Any) -> str:
    if raw_round is None:
        return "unknown"
    round_text = str(raw_round).strip().lower()
    if not round_text:
        return "unknown"
    if "qual" in round_text:
        return "qualifier"
    if "final" in round_text and "semi" not in round_text:
        return "final"
    if "semi" in round_text:
        return "semi_final"
    if "quarter" in round_text:
        return "quarter_final"
    if "group" in round_text:
        return "group_stage"
    if round_text.startswith("r") or "round" in round_text:
        return "early_round"
    return "other"


class SnapshotCSVAdapter(LocalCSVAdapter):
    """In-memory LocalCSVAdapter view frozen at a single prediction cutoff."""

    def __init__(
        self,
        *,
        players_df: pd.DataFrame,
        matches_df: pd.DataFrame,
        cutoff: pd.Timestamp,
        timestamp_column: str | None = None,
        laplace_alpha: float = 2.0,
    ) -> None:
        self.players_path = Path("<memory_players>")
        self.matches_path = Path("<memory_matches>")
        self.laplace_alpha = laplace_alpha
        self._players_df = players_df.copy().reset_index(drop=True)
        self._players_lookup_df = None
        self._player_params_cache: dict[tuple[str, int, str | None, str], dict[str, Any]] = {}
        self._head_to_head_cache: dict[tuple[str, str, int, str | None], dict[str, Any]] = {}
        self._global_prior_cache: dict[str | None, dict[str, Any]] = {}
        self._influence_weights_cache = None
        self.snapshot_cutoff = pd.Timestamp(cutoff)
        self.timestamp_column = timestamp_column if timestamp_column in matches_df.columns else None

        frozen_matches = matches_df.copy()
        frozen_matches["date"] = pd.to_datetime(frozen_matches["date"], utc=False)
        if self.timestamp_column is not None:
            frozen_matches[self.timestamp_column] = pd.to_datetime(frozen_matches[self.timestamp_column], utc=False)
            mask = frozen_matches[self.timestamp_column] <= self.snapshot_cutoff
            sort_columns = [self.timestamp_column, "date"]
        else:
            mask = frozen_matches["date"] <= self.snapshot_cutoff.normalize()
            sort_columns = ["date"]

        self._matches_df = frozen_matches.loc[mask].sort_values(sort_columns).reset_index(drop=True)


def create_snapshot_adapter(
    *,
    players_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    timestamp_column: str | None = None,
    laplace_alpha: float = 2.0,
) -> SnapshotCSVAdapter:
    return SnapshotCSVAdapter(
        players_df=players_df,
        matches_df=matches_df,
        cutoff=cutoff,
        timestamp_column=timestamp_column,
        laplace_alpha=laplace_alpha,
    )


def fetch_and_build_dataset(
    *,
    working_dir: str | Path,
    players_out: str | Path,
    matches_out: str | Path,
) -> HistoricalDatasetPaths:
    scraper = ShuttleSetScraper()
    work_dir = Path(working_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    set_dir = scraper.fetch_set_dir(work_dir)
    return build_historical_dataset(
        raw_set_dir=set_dir,
        players_out=players_out,
        matches_out=matches_out,
    )
