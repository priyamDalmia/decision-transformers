import logging
import logging.config
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dt.paths import LOGGING_CONFIG, PROJECT_ROOT


# Single shot setup for Python logging module.
def setup_logging(
    config_path: str | Path = LOGGING_CONFIG,
    overrides: Mapping[str, Any] | None = None,
) -> None:
    """
    Load logging config with OmegaConf and apply it via dictConfig -- once.

    Call this from entry points only (scripts, __main__ blocks) so that
    importing library code never reconfigures logging. Callers should get their
    own logger with logging.getLogger(__name__) afterwards.

    Example usage:
        def main() -> None:
            setup_logging(
                    overrides={
                        "root": {
                            "handlers": ["stderr", "file-stderr"]
                        }
                    }
            )
            logger = logging.getLogger(__name__)
    """

    cfg = OmegaConf.load(config_path)

    if overrides is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(overrides)))

    # dictconfig wants a plain dict; resove ${oc.env:} now.
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # Anchor relative log paths to the project root so logs land in the same
    # place no matter which directory a script is run from. Then ensure the
    # parent dir exists for every file-based handler.
    for handler in cfg_dict.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            path = Path(filename)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            handler["filename"] = str(path)
            path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(cfg_dict)
