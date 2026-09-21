"""Full-paper experiment orchestration for CrossFM-Align.

The modules in this package deliberately keep heavyweight model imports lazy.  This
allows protocol validation, task enumeration, sharding, and artifact verification
to run on a CPU-only development machine before consuming GPU quota.
"""

from .protocol import ExperimentTask, load_protocol, tasks_for_profile

__all__ = ["ExperimentTask", "load_protocol", "tasks_for_profile"]

