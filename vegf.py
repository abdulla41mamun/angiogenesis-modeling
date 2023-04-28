import numpy as np
import random

from typing import List, Tuple, Union

np.random.seed(42)
random.seed(42)


class VEGFMatrix:
    def __init__(self, grid_size: Union[int, Tuple[int, int]], growth_vector: Union[float, List[float]]):
        """
        Instantiate the VEGF matrix.

        Parameters
        ----------
        grid_size: Union[int, Tuple[int, int]]
            Size of the internal matrix. If integer is sent, the squared matrix is instantiated.
        growth_vector: Union[float, List[float]]
            Growth vector. If float value received, a vector with matching shape for the matrix is instantiate.
        diffusion_factor: float
            Diffusion factor that should be applied to the growth vector. By default, 1.0 - no diffusion is applied.

        """
        self.height, self.width = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
        self.shape = (self.height, self.width)
        self.matrix = np.zeros(self.shape)
        self.diffusion_factor = 0.1
        self.p_move = 0.6
        self.add_growth_vector(growth_vector=growth_vector)

    def add_growth_vector(self, growth_vector: Union[float, List[float]]):
        """
        Add growth vector into the matrix.

        Parameters
        ---------
        growth_vector: Union[float, List[float]]
            Growth vector. If float value received, a vector with matching shape for the matrix is instantiate.

        """
        if isinstance(growth_vector, float):
            growth_vector = [growth_vector] * self.width

        if len(growth_vector) != self.width:
            raise ValueError(f'Growth vector should match matrix shape. Expected length {self.width}.')

        self.matrix[0, :] += np.array(growth_vector)

    def neighbors(self, radius, i, j):
        neighborhood = np.ones((self.height, self.width)) * -np.inf
        i_slice = slice(max(0, i - radius), min(i + radius + 1, self.width))
        j_slice = slice(max(0, j - radius), min(j + radius + 1, self.width))

        neighborhood[i_slice, j_slice] = self.matrix[i_slice, j_slice]
        return neighborhood

    # def diffuse(self):
    #     i = 0
    #     j = 0
    #     center = self.matrix[i][j]
    #     print(center)
    #
    #     gradient_matrix = self.neighbors(1, i, j)
    #     print(gradient_matrix)
    #     gradient_matrix[gradient_matrix != -np.inf] -= center
    #
    #     gradient_matrix[gradient_matrix < 0] *= self.diffusion_factor
    #     print(gradient_matrix)
    #
    #     gradient_matrix = np.abs(gradient_matrix)
    #     print(gradient_matrix)
    #     gradient_matrix[gradient_matrix == np.inf] = 0
    #     print(gradient_matrix)
    #
    #     print(self.matrix)
    #     self.matrix = self.matrix + gradient_matrix
    #     self.matrix[i, j] -= np.sum(gradient_matrix)
    #     print(self.matrix)

    def diffuse(self):
        for i in range(self.height):
            for j in range(self.width):
                center = self.matrix[i][j]
                if center == 0.0:
                    continue

                gradient_matrix = self.neighbors(1, i, j)
                gradient_matrix[gradient_matrix != -np.inf] -= center

                gradient_matrix[gradient_matrix < 0] *= self.diffusion_factor

                gradient_matrix[gradient_matrix == -np.inf] = 0.0
                gradient_matrix[gradient_matrix > 0] = 0.0
                gradient_matrix = np.abs(gradient_matrix)

                self.matrix = self.matrix + gradient_matrix
                self.matrix[i][j] -= np.sum(gradient_matrix)

    def random_diffuse(self):
        tmp_matrix = np.zeros(self.matrix.shape)
        for i in range(self.height):
            for j in range(self.width):
                center = self.matrix[i,j]
                if center == 0.0:
                    continue
                
                rand_vector = np.random.rand(int(center),1)
                mask_move = rand_vector < self.p_move
                tmp_matrix[i,j] -= np.sum(mask_move)

                moved_particles = rand_vector[mask_move]
                if i+1 < self.height:
                    tmp_matrix[i+1,j]  += np.sum(np.logical_and(moved_particles>=0, moved_particles<.25*self.p_move))
                else:
                    tmp_matrix[i,j]  += np.sum(np.logical_and(moved_particles>=0, moved_particles<.25*self.p_move))
                if i-1 >= 0:
                    tmp_matrix[i-1,j]  += np.sum(np.logical_and(moved_particles>=.25*self.p_move, moved_particles<.50*self.p_move))
                else:
                    tmp_matrix[i,j]  += np.sum(np.logical_and(moved_particles>=.25*self.p_move, moved_particles<.50*self.p_move))
                if j+1 < self.width:
                    tmp_matrix[i,j+1]  += np.sum(np.logical_and(moved_particles>=.50*self.p_move, moved_particles<.75*self.p_move))
                else:
                    tmp_matrix[i,j]  += np.sum(np.logical_and(moved_particles>=.50*self.p_move, moved_particles<.75*self.p_move))
                if j-1 >= 0:
                    tmp_matrix[i,j-1]  += np.sum(np.logical_and(moved_particles>=.75*self.p_move, moved_particles<1.0*self.p_move))
                else:
                    tmp_matrix[i,j]  += np.sum(np.logical_and(moved_particles>=.75*self.p_move, moved_particles<1.0*self.p_move))
                # gradient_matrix = self.neighbors(1, i, j)
                # gradient_matrix[gradient_matrix != -np.inf] -= center

                # gradient_matrix[gradient_matrix < 0] *= self.diffusion_factor

                # gradient_matrix[gradient_matrix == -np.inf] = 0.0
                # gradient_matrix[gradient_matrix > 0] = 0.0
                # gradient_matrix = np.abs(gradient_matrix)

                # self.matrix = self.matrix + gradient_matrix
                # self.matrix[i][j] -= np.sum(gradient_matrix)
        # tmp_matrix = tmp_matrix * 0.1
        self.matrix = np.add(self.matrix, tmp_matrix)

    def __str__(self) -> str:
        """Stringify matrix for printing."""
        return np.array_str(self.matrix)

    def __call__(self, i: int, j: int) -> float:
        """Return element [i, j] from internal matrix."""
        return self.matrix[i, j]

    def consume(self, i: int, j: int, value: Union[int, float], neighborhood_size: int = 0):
        """
        Consume VEGF from (neighborhood of) specified position.

        Parameters
        ---------
        i: int
            Specifies a row in the matrix.
        j: int
            Specified a column in the matrix.
        value: Union[int, float]
            Value to be consumed.
        neighborhood_size: int
            Size of the neighborhood to be affected by consumption. By default, 0 - the consumption is applied only
            at the specified position.

        """
        i_slice = slice(max(0, i - neighborhood_size), min(i + neighborhood_size + 1, self.width))
        j_slice = slice(max(0, j - neighborhood_size), min(j + neighborhood_size + 1, self.width))
        self.matrix[i_slice, j_slice] -= float(value)
        self.matrix[self.matrix < 0.0] = 0.0

    def clip(self, clip_threshold: float):
        self.matrix[self.matrix < clip_threshold] = 0.0

if __name__ == "__main__":
    vegf_matrix = VEGFMatrix(grid_size=4, growth_vector=1000.0)
    # Print matrix.
    print(f'Matrix:\n{vegf_matrix}')
    # Matrix shape.
    print(f'Matrix shape: {vegf_matrix.shape}')
    # Get element.
    print(f'Element [1, 2]: {vegf_matrix(1, 2)}')
    # Consume vegf - square neighborhood:
    vegf_matrix.consume(1, 2, value=0.1, neighborhood_size=0)
    print(f'Matrix after consuming VEGF:\n{vegf_matrix}')

    vegf_matrix.diffuse()
