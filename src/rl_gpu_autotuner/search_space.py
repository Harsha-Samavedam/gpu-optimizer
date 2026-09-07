from __future__ import annotations

from itertools import product

from .domain import ScheduleConfig, Workload


def configs_for(workload: Workload) -> list[ScheduleConfig]:
    del workload  # Later: condition choices on kernel type/dimensions.
    configs = [
        ScheduleConfig(m, n, k, warps, stages)
        for m, n, k, warps, stages in product(
            (16, 32, 64, 128),
            (16, 32, 64, 128),
            (16, 32, 64),
            (1, 2, 4, 8),
            (1, 2, 3, 4),
        )
    ]
    return [config for config in configs if config.is_legal()]
