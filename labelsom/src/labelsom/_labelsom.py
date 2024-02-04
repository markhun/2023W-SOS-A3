import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

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

        qe_vector = np.sqrt(np.sum(np.power(weights - input_vector, 2), axis=1))

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


def generate_label_matrix_and_hit_histogram(
    m: int,
    n: int,
    weights: npt.ArrayLike,
    input_data: npt.ArrayLike,
    attribute_names: list[str],
    number_of_labels_to_generate: int,
    ignore_labels_with_zero: bool,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Generate two numpy matrices of dimension m * n for the given SOM data.

    1. A matrix containing the `labelsom.Label`s for each unit.
    2. A hit histogram matrix

    Parameters
    ----------
    m
        Y dimension of the given SOM data.
    n
        X dimension of the given SOM data
    weights
        The weight vectors of the given SOM.
    input_data
        The input vectors for the given SOM.
    attribute_names
        The attribute names to use for labeling.
    number_of_labels_to_generate
        Number of labels to generate per uni.
    ignore_labels_with_zero
        If labels with a mean of 0 should be ignored.

    Returns
    -------
    tuple
        1.  A matrix containing the `labelsom.Label`s for each unit.
        2. A hit histogram matrix
    """

    unit_idx_to_mapped_indices, hit_histogram = (
        generate_unit_idx_to_mapped_indices_mapping(
            m=m,
            n=n,
            weights=weights,
            input_data=input_data,
        )
    )

    label_matrix = np.empty(m * n, dtype=object)

    for unit_idx in range(m * n):
        labels = generate_label_for_unit(
            vec_dim=len(attribute_names),
            mapped_inputs=input_data.take(unit_idx_to_mapped_indices[unit_idx], axis=0),
            weight_vector=weights[unit_idx],
            attribute_names=attribute_names,
            number_of_labels_to_generate=number_of_labels_to_generate,
            ignore_labels_with_zero=ignore_labels_with_zero,
        )
        # print(f"{unit_idx=} : {labels=}")
        label_matrix[unit_idx] = labels

    return label_matrix.reshape(m, n), hit_histogram


def pretty_print_label_matrix(
    label_matrix: npt.ArrayLike,
    hit_histogram: npt.ArrayLike,
    include_mean: bool,
    include_quantization_error: bool,
):
    """Print a labeling matrix and the corresponding hit histogram data as an HTML table."""

    def _pretty_print_lables(lables: list[Label]):
        """Helper function to style table cells"""
        out = "<table class='SOMlabeling'>"

        for label in lables:
            mean_td = (
                "<td class='mean'>m:{:.2f}</td>".format(label.mean)
                if include_mean
                else ""
            )
            qe_td = (
                "<td class='qe'>qe:{:.2f}</td>".format(label.quantization_error)
                if include_quantization_error
                else ""
            )

            out += "<tr>"
            out += "<td>" + str(label.label) + "</td>" + mean_td + qe_td
            out += "</tr>"

        out += "</table>"
        return out

    def _style_label_table(styler):
        """Styling function, which matches signature to be passed to Panda's `style.pipe()`"""
        styler.format(_pretty_print_lables)
        styler.set_properties(**{"text-align": "left"})
        styler.set_table_styles(
            [
                {"selector": ".mean", "props": [("text-align", "left")]},
                {"selector": ".qe", "props": [("text-align", "left")]},
                {
                    "selector": "tr",
                    "props": [("background", "none")],
                },  # Needed to overwrite jupyter CSS
            ]
        )
        styler.background_gradient(axis=None, cmap="Reds", gmap=hit_histogram)
        return styler

    label_table = pd.DataFrame(label_matrix)
    return label_table.style.pipe(_style_label_table)


def write_labelsom_to_file(
    label_matrix: npt.ArrayLike,
    hit_histogram: npt.ArrayLike,
    include_mean: bool,
    include_quantization_error: bool,
    directory_to_write_file_to: Path,
    file_name=None,
):
    directory_to_write_file_to.mkdir(parents=True, exist_ok=True)

    if file_name:
        out_file = directory_to_write_file_to / f"{file_name}.html"
    else:
        out_file = directory_to_write_file_to / "labelsom.html"

    styler = pretty_print_label_matrix(
        label_matrix,
        hit_histogram,
        include_mean,
        include_quantization_error,
    )

    styler.set_table_styles(
        [
            {
                "selector": ".SOMLabeling",
                "props": [
                    ("background", "inherit"),
                    ("background-clip", "text"),
                    ("-webkit-background-clip", "text"),
                    ("color", "#000000"),
                    ("mix-blend-mode", "darken"),
                ],
            },  # Ensures that labeling text is readable even on cells with darker background
        ]
    )

    html = styler.to_html()

    out_file.write_text(html)

    return out_file
