# utils/attack_success.py
"""
Placeholder attack success scoring.

REQUIREMENT:
- Temporary stand-in for an LLM judge.
- Score is an integer in {0,1,2,3}
- Generated via seeded RNG (no regex, no interpretation).
"""

import random
from typing import Optional
from dataclasses import dataclass


@dataclass
class AttackSuccessScorer:
    seed: int = 0

    def score(
        self,
        pair_id: str,
        layer: str,
        step: int,
        prompt_harm: str,
        prompt_benign: str,
        generated_text: Optional[str] = None,
    ) -> int:
        """
        Returns a deterministic pseudo-random score in {0,1,2,3} based on inputs.

        Note: generated_text is accepted only for logging parity; it is NOT used
        semantically and does not affect scoring unless included in the hash seed
        (we do not include it, to avoid accidental coupling to output content).
        """
        # Deterministic mixing
        key = f"{self.seed}|{pair_id}|{layer}|{step}|{prompt_harm}|{prompt_benign}"
        local = random.Random(key)
        return local.randint(0, 3)