"""Allow ``python -m llm_eval_kit``."""

import sys

from .cli import main

sys.exit(main())
