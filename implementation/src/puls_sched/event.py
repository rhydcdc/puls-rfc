from dataclasses import dataclass, field
from enum import Enum, auto


class EventType(Enum):
    KERNEL_COMPLETION = auto()
    REQUEST_ARRIVAL = auto()
    ADMISSION_TICK = auto()


@dataclass
class Event:
    timestamp: float
    type: EventType
    payload: dict = field(default_factory=dict)
