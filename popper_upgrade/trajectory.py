from dataclasses import dataclass
from typing import Any


@dataclass
class Trajectory:
    log: dict[str, list[str]]
    tracked_tests: list[str]
    tracked_stat: list[float]
    res: bool
    res_stat: float
    parsed_result: dict[str, Any]
    last_message: str

    @classmethod
    def from_popper(cls, popper_instance, validate_result):
        agent = popper_instance.agent
        return cls(
            log=agent.log,
            tracked_tests=agent.tracked_tests,
            tracked_stat=agent.tracked_stat,
            res=agent.res,
            res_stat=agent.res_stat,
            parsed_result=validate_result["parsed_result"],
            last_message=validate_result["last_message"],
        )
