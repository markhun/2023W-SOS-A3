import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


@dataclass
class Label:
    label: str
    mean: float
    quantization_error: float

    def __repr__(self):
        return f"{self.label}"


def generate_unit_idx_to_mapped_indices_mapping(m, n, weights, input_data):
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

    return unit_i_to_mapped_inputs, hit_histogram.reshape(m, n)


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


def generate_label_for_unit(
    vec_dim: int,
    mapped_inputs: npt.ArrayLike,
    weight_vector: npt.ArrayLike,
    attribute_names: list[str],
    number_of_labels_to_generate: int,
    ignore_labels_with_zero: bool,
) -> list[Label]:
    """Generate the label data for the inputs of a given unit

    Parameters
    ----------
    vec_dim
        Dimension of input vectors.
    mapped_inputs
        Input vectors mapped to the unit for which to generate labels.
    weight_vector
        Weights for the given unit.
    attribute_names
        Attribute names to use for labeling.
    number_of_labels_to_generate
        Number of labels to generate. Can be at maximum `vec_dim`.
    ignore_labels_with_zero
        If labels with a mean of 0 should be ignored.

    Returns
    -------
        A list of labels generated depending of the passed input data.

    Raises
    ------
    RuntimeError
        If `vec_dim` is not equal to passed number of attribute names.
    """
    if not vec_dim == len(attribute_names):
        raise RuntimeError("Data dimension must match number of attributes !")

    if len(mapped_inputs) == 0:
        return []
    if number_of_labels_to_generate > vec_dim:
        number_of_labels_to_generate = vec_dim

    means = np.mean(mapped_inputs, axis=0)
    quantization_errors = np.mean(np.absolute(mapped_inputs - weight_vector), axis=0)

    labels = []
    for attribute_idx in range(vec_dim):
        mean = means[attribute_idx]
        quantization_error = quantization_errors[attribute_idx]

        # ignoreLabelsWithZero criteria from ``LabelSOM.java``:
        #
        # if we shall ignore zero labels, ignore those with mean==0, and very small qe
        if ignore_labels_with_zero and mean == 0 and quantization_error * 100 < 0.1:
            labels.append(Label("", mean, quantization_error))
        else:
            labels.append(
                Label(attribute_names[attribute_idx], mean, quantization_error)
            )

    labels = _select_top_labels(labels, number_of_labels_to_generate, vec_dim)

    # LabelSOM.java executes:
    #
    #       Label.sortByValueQe(labels, Label.SORT_DESC, Label.SORT_ASC);
    #
    # which is kinda hard to mimic in Python
    labels = sorted(labels, key=lambda label: (-label.mean, label.quantization_error))

    return labels
