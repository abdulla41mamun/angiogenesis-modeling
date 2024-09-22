from cell import CellMatrix
from vegf import VEGFMatrix

from endothelial import endothelial_net
from helpers import consume_vegf, update_tokens, update_cell_types
from plot_util import plot_cell_matrix, plot_vegf_matrix, count_cells, plot_cell_counts
from datetime import datetime


HEIGHT, WIDTH = 11, 15
INIT_GROWTH_VECTOR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

VEGF_clip = True
CLIP_THRESHOLD = 1e-4

RADIUS = 1
PARAMETERS = {
    'alpha_P': 0.27067,
    'alpha_V': 0.05,  # VEGF concentration threshold for the Endothelial to Tip cell transition # Changed because P was not increaseing and S and T were not showing
    'radius': RADIUS,  # radius of the cell
    'd_min': 5,  # estimate, based on 8*R_A (tumor action radius), R_A = 1.214*R, where R = tumor cell radius
    'd_PS': 1.55 * RADIUS,  # minimal distance from tip cell for S -> E transition
    'd_SP': 1.55 * RADIUS,  # maximal distance from tip cell for E -> S transition
    'S_delay': 2,
    'SS_delay': 1
}

TIME_STAMP = str(datetime.now().strftime("%Y-%m-%d_%H%M"))
RANDOM_DIFFUSION = True

if __name__ == "__main__":

    cell_matrix = CellMatrix(grid_size=(HEIGHT, WIDTH))
    vegf_matrix = VEGFMatrix(grid_size=(HEIGHT, WIDTH), growth_vector=INIT_GROWTH_VECTOR)

    n, transitions, places = endothelial_net(
        cell_matrix=cell_matrix,
        vegf_matrix=vegf_matrix,
        params=PARAMETERS
    )

    if RANDOM_DIFFUSION:
        file_name = TIME_STAMP + '_random'
    else:
        file_name = TIME_STAMP


    cells_per_iteration = []
    for i in range(0, 201):

        print(f'Clock: {i}')
        # print(f'\tMarkings: {n.get_marking()}')
        print(cell_matrix)

        # # STEP 0: PLOTS of cell matrix and VEGF matrix
        # plot cell matrix
        plot_cell_matrix(cell_matrix=cell_matrix, iteration=i, file_name=file_name)
        # plot VEGF matrix
        plot_vegf_matrix(vegf_matrix=vegf_matrix, iteration=i, file_name=file_name)
        # counts cells in cell matrix
        cells_per_iteration.append(count_cells(cell_matrix))

        # STEP 1: consume VEGF.
        consume_vegf(cell_matrix, vegf_matrix)

        # STEP 2: diffuse VEGF.
        if RANDOM_DIFFUSION:
            vegf_matrix.random_diffuse()
        else:
            vegf_matrix.diffuse()

        # STEP 2.1: round low VEGF values to zero.
        if VEGF_clip:
            vegf_matrix.clip(CLIP_THRESHOLD)

        # STEP 3: update tokens accordingly.
        for place_name in places:
            tokens = update_tokens(
                tokens=n.place(place_name).tokens,
                cell_matrix=cell_matrix,
                vegf_matrix=vegf_matrix
            )
            n.place(place_name).reset(tokens)

        # STEP 4: transition cells.
        for transition_name in transitions:
            modes = n.transition(transition_name).modes()

            while len(modes) > 0:
                mode = modes.pop()
                enabled = n.transition(transition_name).enabled(mode)
                token = mode.dict()['t']

                if enabled:
                    if cell_matrix.execute_transition(transition_name=transition_name, mode=mode):
                        n.transition(transition_name).fire(mode)

                        if transition_name == 'S-SS':
                            t = {place_name: n.place(place_name).tokens for place_name in places}
                            position = mode.dict()['t'][:2]

                            tokens = cell_matrix.divide_cell(position=position, tokens=t, places=places)
                            if len(tokens) > 0:
                                for place_name, cell_type in places.items():
                                    n.place(place_name).reset(tokens[place_name])
                                # New stalk cells have the time 0, so they will not be considered for this clock.
                                modes = n.transition(transition_name).modes()

                        # STEP 4.1 update cell types in the matrix according to the transitions
                        # print(f'Token: {token} updated to {places[transition_name[-1]]}.')
                        update_cell_types(tokens=[token],
                                          cell_matrix=cell_matrix,
                                          cell_type=places[transition_name[-1]])
                    continue


        # STEP 6: move tip cells.
        tip_cell_tokens = n.place('T').tokens
        tokens = cell_matrix.move_tip_cells(tip_cell_tokens)
        n.place('T').reset(tokens)

        # print('Exit matrix')
        # print(cell_matrix)
        # print('-----')

    # Plot number of cells per iteration
    plot_cell_counts(cells_per_iteration, file_name=file_name)
