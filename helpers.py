from cell import Cell, CellMatrix, CellType
from vegf import VEGFMatrix

from typing import List, Tuple


def consume_vegf(
        cell_matrix: CellMatrix,
        vegf_matrix: VEGFMatrix
):
    """Consume the necessary VEGF for each position."""
    height, width = cell_matrix.shape
    for i in range(height):
        for j in range(width):
            if type(cell_matrix(i, j)) == Cell:
                vegf_matrix.consume(i, j, cell_matrix(i, j).consumption_rate())


def get_tokens(
        place_name: str,
        cell_matrix: CellMatrix,
        vegf_matrix: VEGFMatrix,
        cell_type: CellType
) -> List[Tuple[int, int, float, float, int]]:
    """Return tokens for the PetriNet. Format: (row, col, dist to tip cell, VEGF level)"""
    height, width = cell_matrix.shape
    tokens = []
    if place_name == 'SS':
        return tokens

    for i in range(height):
        for j in range(width):
            if type(cell_matrix(i, j)) == Cell and cell_matrix(i, j).cell_type == cell_type:
                dist = cell_matrix.distance_to_nearest_tip_cell(position=(i, j))
                vegf = vegf_matrix(i, j)
                tokens.append((i, j, dist, vegf, 0))
    return tokens


def update_tokens(
        tokens: List[Tuple[int, int, float, float, int]],
        cell_matrix: CellMatrix,
        vegf_matrix: VEGFMatrix
) -> List[Tuple[int, int, float, float, int]]:
    """Update the distance to the nearest tip cell and the VEGF level for each token in the input list."""
    updated = []
    for (i, j, _, _, time) in tokens:
        dist = cell_matrix.distance_to_nearest_tip_cell(position=(i, j))
        vegf = vegf_matrix(i, j)
        updated.append((i, j, dist, vegf, time))
    return updated


def update_cell_types(
        tokens: List[Tuple[int, int, float, float, int]],
        cell_matrix: CellMatrix,
        cell_type
):
    """Update the distance to the nearest tip cell and the VEGF level for each token in the input list."""
    for (i, j, _, _, _) in tokens:
        cell_matrix.cell_transition(position=(i, j), cell_type=cell_type)

