# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""``domyn-bfcl``: upstream's ``bfcl`` CLI with domyn_bfcl installed.

Accepts every ``bfcl`` command and option, plus ``--generation-config '<json>'`` (any
command): the run's generation settings from domyn-eval. Domyn handlers read all of them;
upstream's handlers get its ``temperature`` through ``generate --temperature``.

``$API_KEY`` is the endpoint's key when ``$REMOTE_OPENAI_API_KEY`` is unset, so the caller
never has to copy the secret onto a command line.
"""

import json
import os
import sys

from domyn_bfcl import config
from domyn_bfcl.install import install


def pop_generation_config(argv: list[str]) -> tuple[dict, list[str]]:
    """Split ``--generation-config <json>`` (or ``=<json>``) out of a ``bfcl`` argv."""
    rest, generation_config = [], {}
    args = iter(argv)
    for arg in args:
        if arg == "--generation-config":
            generation_config = json.loads(next(args))
        elif arg.startswith("--generation-config="):
            generation_config = json.loads(arg.split("=", 1)[1])
        else:
            rest.append(arg)
    return generation_config, rest


def pass_temperature(args: list[str], temperature: float | None) -> list[str]:
    """Add ``--temperature`` to a ``generate`` argv, for the handlers that read only that.

    Unset, upstream's default applies. A ``--temperature`` already given must match.
    """
    if temperature is None or not args or args[0] != "generate":
        return args
    given = []
    for i, arg in enumerate(args):
        if arg == "--temperature" and i + 1 < len(args):
            given.append(args[i + 1])
        elif arg.startswith("--temperature="):
            given.append(arg.split("=", 1)[1])
    if not given:
        return [*args, "--temperature", str(temperature)]
    if any(float(value) != temperature for value in given):
        msg = f"--temperature {given[-1]} conflicts with --generation-config temperature {temperature}"
        raise SystemExit(msg)
    return args


def use_api_key() -> None:
    """Fall back to ``$API_KEY`` for upstream's ``$REMOTE_OPENAI_API_KEY``."""
    if api_key := os.getenv("API_KEY"):
        os.environ.setdefault("REMOTE_OPENAI_API_KEY", api_key)


def main(argv: list[str] | None = None) -> None:
    """Run upstream's ``bfcl`` CLI with the Domyn models, categories and patches installed."""
    use_api_key()
    generation_config, args = pop_generation_config(sys.argv[1:] if argv is None else argv)
    install(generation_config)
    args = pass_temperature(args, config.GENERATION.temperature)
    from bfcl_eval.__main__ import cli

    cli(args=args, prog_name="domyn-bfcl")


if __name__ == "__main__":
    main()
