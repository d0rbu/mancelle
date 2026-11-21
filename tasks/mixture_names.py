"""SFT mixture names shared by the task builders and the feature-path tables.

A leaf module on purpose: ``steering.feature_paths`` needs the steered
mixture set to spell out recovery commands, and importing ``tasks.mixtures``
there would pull every HuggingFace task module into a ubiquitous path
utility.
"""

from enum import StrEnum

from beartype import beartype


class SFTMixture(StrEnum):
    """Available SFT training mixtures."""

    STANDARD = "standard"
    CAPABILITIES = "capabilities"
    SAFETY = "safety"
    CHATCORE = "chatcore"
    STANDARD_STEERED = "standard_steered"
    STANDARD_STEERED_BALANCED = "standard_steered_balanced"


# Mixtures that splice a steered JSONL into the standard train tasks.  They
# consume the standard knobs plus the JSONL path, and validate against
# STANDARD's own val split (see ``sft_val_family``) rather than minting their
# own val phase.
STEERED_MIXTURES: frozenset[SFTMixture] = frozenset(
    {SFTMixture.STANDARD_STEERED, SFTMixture.STANDARD_STEERED_BALANCED}
)


@beartype
def sft_val_family(mixture: SFTMixture) -> SFTMixture:
    """Return the mixture whose validation artifacts ``mixture`` validates on."""
    return SFTMixture.STANDARD if mixture in STEERED_MIXTURES else mixture
