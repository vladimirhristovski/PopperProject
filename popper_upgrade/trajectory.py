from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Trajectory:
    log: Dict[str, List[str]]
    tracked_tests: List[str]
    tracked_stat: List[float]
    res: bool
    res_stat: float
    parsed_result: Dict[str, Any]
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
