import logging
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
LOGGER = logging.getLogger(__name__)

@dataclass
class Label:
    label: str
    mean: float
    quantization_error: float


def generate_unit_idx_to_mapped_indices_mapping(n, m, weights, input_data):
    """Find mapped units similar to `PySOMVis/coding_assigment:HitHist` implementation

    Using mapping function provided by https://www.ifs.tuwien.ac.at/~andi/somlib/labelsom.html
    """

    hit_histogram = np.zeros(m * n)
    unit_i_to_mapped_inputs = defaultdict(list)
    unit_i_to_qe_vector = {}

    for input_datum_i, input_vector in enumerate(input_data):
        qe_vector = np.sum(np.sqrt(np.power(weights - input_vector, 2)), axis=1)

        unit_index = np.argmin(qe_vector)
        unit_index = int(unit_index)  # cast np.int32 datatype to python integer

        hit_histogram[unit_index] += 1
        unit_i_to_mapped_inputs[unit_index].append(input_datum_i)

    return unit_i_to_mapped_inputs, hit_histogram

