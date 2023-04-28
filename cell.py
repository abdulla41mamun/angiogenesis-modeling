import numpy as np

from enum import Enum
from typing import Tuple, Union, List, Dict


class CellType(Enum):
    PHALANX = 1,
    STALK = 2,
    TIP = 3,


class Cell(object):
    """
        Cell class representing every cell in the matrix
        -- first scratch

    """

    def __init__(self, cell_type: CellType):
        self.cell_type = cell_type  # PHALANX/TIP/STALK
        self.inner_VEGF: float = 0
        self.consumption_lambda: float = 10

    def transition(self, cell_type: CellType):
        self.cell_type = cell_type

    def __str__(self):
        return str(self.cell_type)

    def consumption_rate(self):
        return self.consumption_lambda


class CellMatrix:
    def __init__(self, grid_size: Union[int, Tuple[int, int]]):
        self.height, self.width = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
        self.shape = (self.height, self.width)
        self.matrix = np.zeros(self.shape,  dtype=object)

        # Initialize the bottom line of the matrix with PHALANX cells.
        for idx in range(self.width):
            self.matrix[self.height - 1, idx] = Cell(CellType.PHALANX)

        # Specifies if there is any tip cell in the cell matrix.
        # Helpful when calculating distances to nearest tip cell.
        self._tip_cells = False
        # self._to_divide = {}

    def cell_transition(self, position: Tuple[int, int], cell_type: CellType):
        """Apply cell type transition."""
        if self.no_cell(position):
            raise ValueError(f'Position {position} does not hold a cell.')
        self.matrix[position[0], position[1]].transition(cell_type)
        self._tip_cells |= cell_type == CellType.TIP

    def no_cell(self, position: Tuple[int, int]) -> bool:
        """Check if the position contains a Cell or it is empty."""
        return self.matrix[position[0], position[1]] == 0

    def _cell_move(self, position: Tuple[int, int], new_position: Tuple[int, int]):
        """Move cell to new position."""
        if self.no_cell(position):
            return
        self.matrix[new_position[0], new_position[1]] = self.matrix[position[0], position[1]]
        self.matrix[position[0], position[1]] = 0

    def move_up(self, cell_position: Tuple[int, int]) -> tuple:
        """Move cell left up, one space, if possible."""
        i, j = cell_position
        if i - 1 < 0:   # Check if the move can be done.
            return i, j
        self._cell_move(position=cell_position, new_position=(i - 1, j))
        return i-1, j

    def move_down(self, cell_position: Tuple[int, int]) -> bool:
        """Move cell left to down, one space, if possible."""
        i, j = cell_position
        if i + 1 >= self.height:  # Check if the move can be done.
            return False
        self._cell_move(position=cell_position, new_position=(i + 1, j))
        return True

    def move_right(self, cell_position: Tuple[int, int]) -> bool:
        """Move cell left to the right, one space, if possible."""
        i, j = cell_position
        if j + 1 >= self.width:  # Check if the move can be done.
            return False
        self._cell_move(position=cell_position, new_position=(i, j + 1))
        return True

    def move_left(self, cell_position: Tuple[int, int]) -> bool:
        """Move cell left to the left, one space, if possible."""
        i, j = cell_position
        if j - 1 < 0:  # Check if the move can be done.
            return False
        self._cell_move(position=cell_position, new_position=(i, j - 1))
        return True

    def distance_to_nearest_tip_cell(self, position: Tuple[int, int]) -> float:
        """Calculates distance to the nearest tip cell."""
        dist_tip = np.inf
        # If no tip cell in the matrix, the distance is inf.
        if not self._tip_cells:
            return dist_tip

        for i in range(self.height):  # i ~ rows ~ y
            for j in range(self.width):  # j ~ columns ~ x
                # If we are on the element of interest, we skip.
                if i == position[0] and j == position[1]:
                    continue
                obj = self.matrix[i, j]
                # If object at position is Cell and its cell type is TIP.
                if type(obj) != Cell or obj.cell_type != CellType.TIP:
                    continue
                # Calculate Euclidean distance of coordinates # TODO: Do we want Euclidean distance?
                current_dist = np.sqrt((position[1] - j) ** 2 + (position[0] - i) ** 2)
                dist_tip = min(dist_tip, current_dist)
        return dist_tip

    def execute_transition(self, transition_name: str, mode) -> bool:
        """Check whether the current transition respects necessary constraints."""
        position = mode.dict()['t'][:2]
        obj = self.matrix[position[0], position[1]]
        if type(obj) != Cell:
            raise ValueError(f'Transition {transition_name} tried to be applied on a non-cell element ({position}).')
        if transition_name == 'P-T':
            return self.should_transition_to_tip(token=position)
        if transition_name == 'S-SS':
            return self.division_possible(position=position)
        return True

    def division_possible(self, position: Tuple[int, int]):
        """
        Check if division of specified cell is possible. The cell can be divided only if
        there is space on the column (it should considered already reserved spots as well.
        """
        column = self.matrix[:position[0], position[1]]
        return np.any(column == 0)
        # empty_spots = np.sum(column == 0)
        # if position[1] in self._to_divide:
        #     empty_spots -= len(self._to_divide[position[1]])
        # return empty_spots > 0

    # def queue_division(self, position: Tuple[int, int]):
    #     """Queue division of specific cell, to be applied later."""
    #     if position[1] not in self._to_divide:
    #         self._to_divide[position[1]] = []
    #     self._to_divide[position[1]].append(position)

    @staticmethod
    def get_key(value, dictionary):
        for key in dictionary:
            if dictionary[key] == value:
                return key

    # -> Dict[int, List[Tuple[int, int, float, float, int]]]
    def divide_cell(self, position: Tuple[int, int], tokens: dict, places: dict):
        """Divide specific cell. Division is done only upwards."""
        obj = self.matrix[position[0], position[1]]
        try:
            if type(obj) != Cell or obj.cell_type != CellType.STALK:
                cell_type = None if type(obj) != Cell else obj.cell_type
                raise ValueError(f'Division not available. Position {position} does not hold a stalk cell. ({cell_type})')
        except Exception as e:
            print("Position", position)
            # print(print(self.str_matrix))
            raise e

        divide_position = (position[0] - 1, position[1])
        if self.no_cell(position=divide_position):
            # print(f'No cell on position {divide_position}')
            self.matrix[divide_position[0], divide_position[1]] = Cell(cell_type=CellType.STALK)
            return {}
        else:
            # {'P': [], 'S': []}
            updated_tokens = {place_name: [] for place_name in places}
            for row in range(0, self.height):
                for col in range(0, self.width):
                    current_position = (row, col)

                    obj = self.matrix[current_position[0], current_position[1]]
                    if type(obj) != Cell:
                        continue

                    retrieved_tokens = CellMatrix.retrieve_tokens(tokens=tokens, position=current_position)

                    flatten_tokens = [item for sublist in list(retrieved_tokens.values()) for item in sublist]
                    if len(flatten_tokens) == 0:
                        raise ValueError(f'No tokens retrieved for position: {current_position}')

                    # Case 1: no modifications needed.
                    if col != position[1] or row >= position[0]:
                        if len(retrieved_tokens) > 1 or len(flatten_tokens) > 1:
                            raise ValueError(f'Multiple tokens at general position {current_position}.')
                        place_name = list(retrieved_tokens.keys())[0]
                        token = retrieved_tokens[place_name][0]
                        updated_tokens[place_name].append(token)
                        continue

                    # Case 2: position needs to be updated.

                    # Case 2.1: only one element retrieved - push with one row up.
                    if len(flatten_tokens) == 1:
                        place_name = list(retrieved_tokens.keys())[0]
                        token = list(retrieved_tokens[place_name][0])
                        token[0] -= 1
                        updated_tokens[place_name].append(tuple(token))
                    else:
                        if 'S' in retrieved_tokens and len(retrieved_tokens['S']) > 1:
                            t = sorted(retrieved_tokens['S'], key=lambda x: x[4])

                            updated_tokens['S'].append(t[0])

                            token = list(t[-1])
                            token[0] -= 1
                            updated_tokens['S'].append(tuple(token))
                        elif 'SS' in retrieved_tokens and len(retrieved_tokens['SS']) > 1:
                            t = sorted(retrieved_tokens['SS'], key=lambda x: x[4])

                            updated_tokens['SS'].append(t[0])

                            token = list(t[-1])
                            token[0] -= 1
                            updated_tokens['SS'].append(tuple(token))
                        elif 'S' in retrieved_tokens and 'SS' in retrieved_tokens:
                            for key in retrieved_tokens:
                                if key == 'SS' and len(retrieved_tokens[key]) > 0:
                                    token = retrieved_tokens[key][0]
                                    updated_tokens[key].append(token)
                                elif len(retrieved_tokens[key]) == 1:
                                    token = list(retrieved_tokens[key][0])
                                    token[0] -= 1
                                    updated_tokens[key].append(tuple(token))
                        else:
                            for key in retrieved_tokens:
                                if key in {'S', 'SS'} and len(retrieved_tokens[key]) > 0:
                                    token = retrieved_tokens[key][0]
                                    updated_tokens[key].append(token)
                                elif len(retrieved_tokens[key]) == 1:
                                    token = list(retrieved_tokens[key][0])
                                    token[0] -= 1
                                    updated_tokens[key].append(tuple(token))

            column = self.matrix[:position[0] + 1, position[1]]
            self.matrix[:position[0], position[1]] = column[1:]

        return updated_tokens

    @staticmethod
    def retrieve_tokens(tokens: dict, position: tuple):
        found_tokens = {}
        for cell_type in tokens:
            for token in list(tokens[cell_type]):
                if token[0] == position[0] and token[1] == position[1]:
                    if cell_type not in found_tokens:
                        found_tokens[cell_type] = []
                    found_tokens[cell_type].append(token)
        return found_tokens

    # def divide_cells(self, tokens: dict, i):
    #     """
    #     Divide all cells that were queued for division.
    #     tokens: dict
    #         Format {place_name: list of tokens}
    #     """
    #     # for positions in self._to_divide.values():
    #     #     for position in list(sorted(positions, key=lambda x: x[0])):
    #     #         print(f'Divide position: {position}')
    #     #         self.divide_cell(position=position, tokens=tokens)
    #     #         print(self.str_matrix)
    #     # updated_idxs = {}
    #     for positions in self._to_divide.values():
    #         for position in list(sorted(positions, key=lambda x: x[0])):
    #             print(f'Divide position: {position}')
    #             update = self.divide_cell(position=position, tokens=tokens)
    #             if i == 32:
    #                 print(update)
    #             # for k in update:
    #             #     if k not in updated_idxs:
    #             #         updated_idxs[k] = []
    #             #     updated_idxs += update[k]
    #             # print(self.str_matrix)
    #
    #     # for place_name in updated_idxs:
    #     #     for idx in updated_idxs[place_name]:
    #     #         tokens[place_name][idx][0] -= 1
    #
    #     # Empty the dictionary of queued cells (they are already divided).
    #     self._to_divide = {}
    #
    #     # return tokens

    def move_tip_cells(self, tokens: List[Tuple[int, int, float, float]]):
        """Move all the tip cells that are bordered by stalk cells."""
        new_tokens = []
        for (i, j, d, vegf, time) in tokens:
                obj = self.matrix[i, j]
                new_i = i
                # If object at position is Cell and its cell type is TIP.
                if type(obj) != Cell or obj.cell_type != CellType.TIP:
                    raise TypeError(f"Position {i}, {j} is not a TIP cell")

                left_obj = self.matrix[i, j-1]
                right_obj = self.matrix[i, j+1]
                if type(left_obj) == Cell and left_obj.cell_type == CellType.STALK \
                        and type(right_obj) == Cell and right_obj.cell_type == CellType.STALK:
                    new_i, j = self.move_up((i, j))
                new_tokens.append((new_i, j, d, vegf, time))
        return new_tokens

    def should_transition_to_tip(self, token: Tuple[int, int]):
        """"""
        i, j = token
        if j == 0 or j == self.width - 1:
            return False

        elif j-2 >= 0 and (type(self.matrix[i, j-2]) == Cell and self.matrix[i, j-2].cell_type == CellType.TIP):
            return False

        elif j+2 < self.width and (type(self.matrix[i, j+2]) == Cell and self.matrix[i, j+2].cell_type == CellType.TIP):
            return False

        elif type(self.matrix[i, j+1]) == Cell and self.matrix[i, j+1].cell_type == CellType.TIP:
            return False

        elif type(self.matrix[i, j-1]) == Cell and self.matrix[i, j-1].cell_type == CellType.TIP:
            return False

        else:
            return True

    @property
    def str_matrix(self):
        str_matrix = np.array([['-'] * self.width] * self.height, dtype=object)
        for i in range(self.height):
            for j in range(self.width):
                if not self.no_cell(position=(i, j)):
                    str_matrix[i, j] = str(self.matrix[i, j])[9]
        return np.array_str(str_matrix)

    def __str__(self) -> str:
        """Stringify matrix for printing."""
        return self.str_matrix

    def __call__(self, i: int, j: int) -> Union[float, Cell]:
        """Return element [i, j] from internal matrix."""
        return self.matrix[i, j]


if __name__ == "__main__":
    cell_matrix = CellMatrix(5)
    print('Cell matrix:')
    print(cell_matrix)
    print('------\n')

    # cell_matrix.move_up(cell_position=(4, 1))
    # print('Cell matrix after movement:')
    # print(cell_matrix)
    # print('------\n')

    # print(f'Parameters for cell at position [4, 2]: {cell_matrix.parameters(position=(4, 2))}')
    # print('------\n')

    cell_matrix.cell_transition(position=(4, 3), cell_type=CellType.TIP)
    cell_matrix.cell_transition(position=(4, 4), cell_type=CellType.STALK)
    cell_matrix.cell_transition(position=(4, 2), cell_type=CellType.STALK)

    print('Cell matrix after transition:')
    print(cell_matrix)
    print('------\n')

    cell_matrix.move_tip_cells()
    print(cell_matrix)

    # print(f'Parameters for cell at position [4, 2]: {cell_matrix.parameters(position=(4, 2))}')

    # cell_matrix.divide_cell((4,4))
    # print(cell_matrix)
    #
    # cell_matrix.divide_cell((4,4))
    # print(cell_matrix)
    #
    # cell_matrix.divide_cell((3,4))
    # print(cell_matrix)
    #
    # cell_matrix.divide_cell((3,4))
    # print(cell_matrix)
    #
    # r = cell_matrix.divide_cell((3,4))
    # print(r)
    # print(cell_matrix)

