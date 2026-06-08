"""run_tp2.py - launcher that spawns bench_tp.py across 2 GPUs.

This exists because torchrun is a shell command but jobs-cli runs Python files.
This wrapper invokes torch.distributed.run with the same arguments as
  `torchrun --standalone --nproc_per_node=2 bench_tp.py`
so the user can launch via `python3 run_tp2.py`.

For the 1-GPU baseline, just run `python3 bench_tp.py` directly (no
RANK/WORLD_SIZE env vars -> world_size defaults to 1, no NCCL init).
"""

import os
import sys

# Inject the args torchrun would have parsed.
sys.argv = [
    'torchrun',
    '--standalone',
    '--nnodes=1',
    '--nproc_per_node=2',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_tp.py'),
]

from torch.distributed.run import main
main()
