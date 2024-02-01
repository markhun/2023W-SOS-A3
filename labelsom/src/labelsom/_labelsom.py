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


def _select_top_labels(labels, number_of_labels_to_generate, dim):
    """Method to select `number_of_labels_to_generate` top labels.

    Direct 1:1 re-implementation from ``LabelSOM.Java``.
    """
    labels_sorted_by_mean = sorted(labels, key=lambda label: label.mean)
    labels_sorted_by_qe = sorted(labels, key=lambda label: label.quantization_error)

    # determine select num top labels
    top_labels = [None for _ in range(number_of_labels_to_generate)]
    found = 0
    lab = 0

    while (
        found < number_of_labels_to_generate and lab < dim
    ):  # go through list sorted by qe
        found2 = False
        lab2 = dim - 1
        while not found2 and lab2 >= dim - number_of_labels_to_generate:
            if labels_sorted_by_mean[lab2] == labels_sorted_by_qe[lab]:
                found2 = True
                top_labels[found] = labels_sorted_by_qe[lab]
                found += 1
            lab2 -= 1
        lab += 1

    return top_labels

