import logging
from dataclasses import dataclass
LOGGER = logging.getLogger(__name__)

@dataclass
class Label:
    label: str
    mean: float
    quantization_error: float

