"""
PCSP file generator.

Mirrors generate_pcsp() from Generate_PCSP.py (lines 17-34):
assembles a complete .pcsp file by concatenating:
  1. Variable declarations (var template)
  2. #define p0 N; ... #define pK M; (extracted frequency counts)
  3. Process definitions (model template)

The output file can be loaded directly into PAT for verification.
"""
import os
from typing import List

from config import SportConfig


class PCSPGenerator:
    """Assembles complete .pcsp files from parameters and templates."""

    def generate(
        self,
        config: SportConfig,
        params: List[int],
        variant_name: str,
        entity1: str,
        entity2: str,
        date: str,
        output_dir: str = '.'
    ) -> str:
        """Generate a .pcsp file for a specific matchup.

        Mirrors generate_pcsp() from Generate_PCSP.py exactly:
          1. Read var.txt
          2. Write #define p0 54; ... lines
          3. Read RH_RH.txt (or whichever variant)
          4. Concatenate all three parts
          5. Write to output file

        Args:
            config: Sport configuration
            params: Flat list of integer frequency counts
            variant_name: Model variant (e.g. "RH_RH")
            entity1: First entity name
            entity2: Second entity name
            date: Match date string (YYYY-MM-DD)
            output_dir: Directory for the output .pcsp file

        Returns:
            Absolute path to the generated .pcsp file
        """
        variant = config.get_variant(variant_name)

        # Build output filename (mirrors Generate_PCSP.py lines 20-21)
        safe_e1 = entity1.replace(' ', '-')
        safe_e2 = entity2.replace(' ', '-')
        filename = f"{variant_name}_{date}_{safe_e1}_{safe_e2}.pcsp"
        filepath = os.path.join(output_dir, filename)

        # 1. Read variable declarations
        with open(config.var_template) as f:
            var_lines = f.readlines()

        # 2. Generate #define lines (mirrors lines 27-28)
        define_lines = [f'#define p{i} {p};\n' for i, p in enumerate(params)]

        # 3. Read process definition template
        with open(variant.template_file) as f:
            model_lines = f.readlines()

        # 4. Concatenate and write (mirrors lines 31-33)
        all_lines = var_lines + define_lines + model_lines
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            f.writelines(all_lines)

        return os.path.abspath(filepath)

    def preview_defines(self, params: List[int]) -> str:
        """Return the #define block as a string.

        Useful for showing parameter summaries to the LLM
        without reading the full .pcsp file.
        """
        return '\n'.join(f'#define p{i} {p};' for i, p in enumerate(params))
