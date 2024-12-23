"""Module to parameterise models."""

from pathlib import Path

import pandas as pd
from parameteriser.brenda.v0 import Brenda


def get_km_and_kcat_from_brenda(
    ec: str,
    brenda_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Obtain michaelis and catalytic constants for given ec number.

    You can obtain the database from https://www.brenda-enzymes.org/download.php
    """
    brenda = Brenda()
    if brenda_path is not None:
        brenda.read_database(brenda_path)

    kms, kcats = brenda.get_kms_and_kcats(
        ec=ec,
        filter_mutant=True,
        filter_missing_sequences=True,
    )
    return kms, kcats
