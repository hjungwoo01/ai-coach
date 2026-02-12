"""
Stage 2: PAT model checker interface.

PAT (Process Analysis Toolkit) is a GUI-based desktop application.
This module provides:
  1. Manual mode: generates instructions for the user to run PAT
  2. Result parser: extracts probability from PAT output text
  3. Mock mode: simulates PAT output for testing the LLM pipeline

When/if a PAT CLI becomes available, the run() method can be
extended to invoke it via subprocess.
"""
import os
import re
from typing import Dict, Optional


class PATRunner:
    """Interface to the PAT model checker."""

    def __init__(self, pat_path: Optional[str] = None):
        """
        Args:
            pat_path: Path to PAT executable (None = manual mode).
        """
        self.pat_path = pat_path

    def run(self, pcsp_file: str) -> Dict:
        """Run PAT on a .pcsp file and return results.

        In manual mode: returns step-by-step instructions.
        In auto mode: placeholder for future CLI invocation.

        Returns:
            Dict with keys:
            - status: 'manual' | 'completed' | 'error'
            - pcsp_file: absolute path to input file
            - instructions: (manual mode) user-facing steps
        """
        abs_path = os.path.abspath(pcsp_file)

        if self.pat_path is None:
            return {
                'status': 'manual',
                'pcsp_file': abs_path,
                'instructions': (
                    f"1. Open PAT (Process Analysis Toolkit)\n"
                    f"2. File -> Open -> select: {abs_path}\n"
                    f"3. Go to the Verification tab\n"
                    f"4. Click 'Verify' on the assertion\n"
                    f"5. Note the probability result\n"
                    f"6. Enter the probability when prompted"
                )
            }

        # Future: subprocess-based PAT CLI invocation
        raise NotImplementedError(
            "Automatic PAT execution not yet implemented. "
            "Use pat_path=None for manual mode."
        )

    def parse_result(self, result_text: str) -> float:
        """Parse a PAT probability result string.

        PAT outputs lines like:
          "The Assertion ... is Valid."
          "Min/Max probability = 0.6473"

        Args:
            result_text: Raw text from PAT output

        Returns:
            Probability as float
        """
        match = re.search(r'probability\s*=\s*([0-9.]+)', result_text)
        if match:
            return float(match.group(1))

        # Try parsing as bare float
        try:
            return float(result_text.strip())
        except ValueError:
            raise ValueError(f"Could not parse PAT result: {result_text}")

    def create_mock_result(self, p1_win_prob: float = 0.55) -> Dict:
        """Create a mock PAT result for testing the full pipeline.

        Args:
            p1_win_prob: Simulated win probability for entity 1

        Returns:
            Dict matching the run() output format
        """
        return {
            'status': 'completed',
            'p1_win_prob': p1_win_prob,
            'p2_win_prob': round(1 - p1_win_prob, 4),
            'raw_output': (
                f'The assertion is valid.\n'
                f'Min probability = {p1_win_prob}'
            )
        }
