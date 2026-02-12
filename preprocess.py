"""
Stage 1: Data loading, filtering, and parameter extraction.

This is the Python preprocessing wrapper (glue code) that abstracts
what Generate_PCSP.py does in the tennis pipeline:
  1. Load CSV with sport-specific column names
  2. Filter by entities (players/teams) and a date window
  3. Extract frequency counts per parameter group
  4. Return a flat list of integer parameters for PCSP generation

All sport-specific logic lives in the SportConfig — this code is generic.
"""
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Dict, Optional

from config import SportConfig, VariantConfig


class DataLoader:
    """Loads and caches the sport CSV data.

    Mirrors the CSV loading at lines 139-152 of Generate_PCSP.py.
    Caching avoids reloading large files on every query.
    """

    def __init__(self, config: SportConfig):
        self.config = config
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load CSV with config-defined column names. Cached after first call."""
        if self._df is None:
            self._df = pd.read_csv(
                self.config.data_file,
                names=self.config.csv_columns
            )
        return self._df

    def reload(self) -> pd.DataFrame:
        """Force reload from disk (e.g. if data file changed)."""
        self._df = None
        return self.load()


class ParameterExtractor:
    """Extracts PCSP parameters from filtered data.

    Mirrors get_params() from Generate_PCSP.py but driven by config.
    Each ParameterGroup defines a filter_query and count_queries.
    The extractor applies them and returns integer frequency counts.
    """

    def __init__(self, config: SportConfig):
        self.config = config

    def filter_entity_data(
        self,
        df: pd.DataFrame,
        entity_name: str,
        opponent_name: str,
        date: str
    ) -> pd.DataFrame:
        """Filter dataset to relevant rows for one entity.

        Mirrors lines 109-112 of Generate_PCSP.py:
          data_ply1 = data.query('date>=@prev_date and date<@date
                                  and ply1_name==@ply1_name
                                  and ply2_name==@ply2_name')

        Args:
            df: Full dataset
            entity_name: e.g. "Novak Djokovic"
            opponent_name: e.g. "Daniil Medvedev"
            date: Match date as "YYYY-MM-DD"

        Returns:
            Filtered DataFrame containing only this entity's historical data
        """
        prev_date = (
            pd.to_datetime(date) -
            relativedelta(years=self.config.lookback_years)
        ).strftime('%Y-%m-%d')

        date_col = self.config.date_column
        e1_col = self.config.entity_columns[0]
        e2_col = self.config.entity_columns[1]

        filtered = df.query(
            f'{date_col} >= @prev_date and {date_col} < @date and '
            f'{e1_col} == @entity_name and {e2_col} == @opponent_name'
        )
        return filtered

    def extract_params(
        self,
        df: pd.DataFrame,
        variant: VariantConfig
    ) -> List[int]:
        """Extract parameter counts from filtered data for one entity.

        Mirrors get_params(df, hand) — iterates through parameter groups
        and runs count queries. Returns raw frequency counts (not normalized).
        PAT normalizes via pcase statements.

        Args:
            df: Pre-filtered DataFrame for one entity
            variant: The model variant (determines which parameter groups to use)

        Returns:
            Flat list of integer frequency counts
        """
        all_params = []
        for group in variant.parameter_groups:
            # Apply the group filter (e.g. "shot_type==1 and from_which_court==1")
            group_df = df.query(group.filter_query) if group.filter_query else df

            # Count each outcome
            for outcome_name, query in group.count_queries.items():
                count = len(group_df.query(query))
                all_params.append(count)

        return all_params

    def get_all_params(
        self,
        df: pd.DataFrame,
        entity1: str,
        entity2: str,
        date: str,
        variant: VariantConfig
    ) -> Tuple[List[int], Dict]:
        """Full parameter extraction for a matchup (both entities).

        Mirrors generate_transition_probs() from Generate_PCSP.py:
          1. Filter data for entity1's perspective
          2. Filter data for entity2's perspective
          3. Extract params for both
          4. Concatenate: entity1_params + entity2_params

        Returns:
            Tuple of:
            - Flat list of all params (entity1 then entity2)
            - Metadata dict with match counts
        """
        df1 = self.filter_entity_data(df, entity1, entity2, date)
        df2 = self.filter_entity_data(df, entity2, entity1, date)

        params1 = self.extract_params(df1, variant)
        params2 = self.extract_params(df2, variant)

        metadata = {
            'entity1': entity1,
            'entity2': entity2,
            'date': date,
            'variant': variant.name,
            'entity1_matches': len(df1[self.config.date_column].unique()),
            'entity2_matches': len(df2[self.config.date_column].unique()),
            'entity1_param_count': len(params1),
            'entity2_param_count': len(params2),
            'total_params': len(params1) + len(params2),
        }

        return params1 + params2, metadata
