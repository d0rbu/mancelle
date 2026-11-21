"""Errors raised by the upstream AxBench adversarial evaluation."""

from beartype import BeartypeConf


class AxBenchAdvError(RuntimeError):
    """An AxBench-Adv contract or pipeline stage failed."""


AXBENCH_ADV_BEARTYPE = BeartypeConf(violation_param_type=AxBenchAdvError)


__all__ = ["AXBENCH_ADV_BEARTYPE", "AxBenchAdvError"]
