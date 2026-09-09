"""
Household ADL / meal-prep scenario.

The PFE fiche cites "préparation d'un repas" as an application example.
Current processed merge (HAPT+WISDM) has no cooking labels, so a real
meal classifier cannot be trained here.

This module:
  - documents the gap,
  - exposes a ScenarioSpec for future datasets (e.g. cooking ADLs),
  - provides a *proxy* demo that treats sit/stand transitions in a
    kitchen-like narrative for UI only (not a scientific meal model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ScenarioSpec:
    name: str
    description: str
    required_labels: List[str]
    available: bool
    note: str


MEAL_PREP_SCENARIO = ScenarioSpec(
    name="meal_preparation",
    description="Suivi / anticipation d'activités de préparation de repas",
    required_labels=[
        "chopping", "stirring", "opening_fridge", "washing_dishes", "cooking",
    ],
    available=False,
    note=(
        "Non implémentable sur HAPT/WISDM (pas de labels cuisine). "
        "Nécessite un corpus ADL cuisine (ex. Opportunity, Cooking dataset). "
        "Resté comme scénario d'application dans la fiche / mémoire."
    ),
)


HOUSEHOLD_PROXY = ScenarioSpec(
    name="household_adl_proxy",
    description=(
        "Proxy UI: enchaînements assis/debout/transition comme 'contexte domestique'"
    ),
    required_labels=["sitting_standing", "stand_to_sit", "sit_to_stand"],
    available=True,
    note="Démo narrative uniquement — ne pas présenter comme détection de repas.",
)


def fiche_scenarios() -> Dict[str, ScenarioSpec]:
    return {
        MEAL_PREP_SCENARIO.name: MEAL_PREP_SCENARIO,
        HOUSEHOLD_PROXY.name: HOUSEHOLD_PROXY,
    }


def meal_status() -> str:
    s = MEAL_PREP_SCENARIO
    return f"[{'OK' if s.available else 'UNAVAILABLE'}] {s.name}: {s.note}"
