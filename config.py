"""
Sport configuration definitions.

A SportConfig captures everything needed to run the pipeline for a specific sport:
- CSV schema (column names)
- Entity identification (player/team columns)
- Parameter extraction queries (what to count from the data)
- PCSP template paths (model structure)
- Model variants (e.g. handedness combinations in tennis)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import dataclasses
import json


@dataclass
class ParameterGroup:
    """One group of frequency counts to extract from data.

    In tennis, a group is e.g. "Deuce Court 1st Serve" with counts for
    serve_T_in, serve_body_in, serve_wide_in, winner, error.

    Attributes:
        name: Human-readable label (e.g. "Deuce Court 1st Serve")
        filter_query: Pandas query string to select relevant rows
                      (e.g. "shot_type==1 and from_which_court==1")
        count_queries: Ordered dict of outcome_name -> pandas query string.
                       Each query is applied to the filtered subset and
                       len() of the result becomes one PCSP parameter.
    """
    name: str
    filter_query: str
    count_queries: Dict[str, str]


@dataclass
class VariantConfig:
    """One model variant (e.g. RH_RH in tennis).

    Different variants use different PCSP templates and may have
    different parameter extraction logic (e.g. direction mappings
    differ by handedness).

    Attributes:
        name: Variant identifier used in filenames (e.g. "RH_RH")
        template_file: Path to the PCSP process definition template
        applies_when: Dict of attribute conditions for auto-selection
                      (e.g. {"ply1_hand": "RH", "ply2_hand": "RH"})
        parameter_groups: Ordered list of ParameterGroups for this variant.
                          The order determines parameter numbering (p0, p1, ...).
    """
    name: str
    template_file: str
    applies_when: Dict[str, str] = field(default_factory=dict)
    parameter_groups: List[ParameterGroup] = field(default_factory=list)


@dataclass
class SportConfig:
    """Top-level sport configuration.

    Attributes:
        sport_name: e.g. "tennis", "basketball"
        data_file: Path to the CSV data file
        csv_columns: Ordered list of column names (CSV has no header)
        entity_name: What to call competing units ("player" or "team")
        date_column: Column name for dates
        entity_columns: [entity1_col, entity2_col] for identifying matchups
        lookback_years: How many years of history to use for parameter extraction
        var_template: Path to the PCSP variable declarations file
        variants: List of model variants
        assertion_template: PCSP assertion line
    """
    sport_name: str
    data_file: str
    csv_columns: List[str]
    entity_name: str
    date_column: str
    entity_columns: List[str]
    lookback_years: int
    var_template: str
    variants: List[VariantConfig]
    assertion_template: str

    @classmethod
    def from_json(cls, path: str) -> 'SportConfig':
        """Load a SportConfig from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        variants = []
        for v in data.get('variants', []):
            groups = [ParameterGroup(**g) for g in v.pop('parameter_groups', [])]
            variants.append(VariantConfig(**v, parameter_groups=groups))
        data['variants'] = variants
        return cls(**data)

    def to_json(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    def get_variant(self, name: str) -> VariantConfig:
        """Look up a variant by name. Raises ValueError if not found."""
        for v in self.variants:
            if v.name == name:
                return v
        raise ValueError(
            f"Unknown variant '{name}'. "
            f"Available: {[v.name for v in self.variants]}"
        )

    def auto_select_variant(self, attributes: Dict[str, str]) -> VariantConfig:
        """Auto-select variant based on entity attributes.

        Args:
            attributes: e.g. {"ply1_hand": "RH", "ply2_hand": "RH"}

        Returns:
            First matching VariantConfig, or the first variant as fallback.
        """
        for v in self.variants:
            if all(attributes.get(k) == val for k, val in v.applies_when.items()):
                return v
        return self.variants[0]
