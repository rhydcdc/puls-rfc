"""puls_sched module entry — `python -m puls_sched` (Impl-9 Q2)."""

import sys

from puls_sched.run import Run


if __name__ == "__main__":
    sys.exit(Run.main(sys.argv[1:]))
