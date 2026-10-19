import copy
from dataclasses import asdict, dataclass
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
            log=copy.deepcopy(agent.log),
            tracked_tests=list(agent.tracked_tests),
            tracked_stat=list(agent.tracked_stat),
            res=agent.res,
            res_stat=agent.res_stat,
            parsed_result=copy.deepcopy(validate_result["parsed_result"]),
            last_message=validate_result["last_message"],
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**copy.deepcopy(d))
