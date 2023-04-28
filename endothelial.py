# Endothelial cell from Fig 3 of the paper
import tpn
import snakes.plugins
# snakes.plugins.load(tpn, "snakes.nets", "snk")

from cell import CellType
from helpers import get_tokens
# from snk import *
from snakes.nets import *
# from tpn import *


def endothelial_net(cell_matrix, vegf_matrix, params: dict):
    net = PetriNet('Endothelial cell transitions')

    # Add init places.
    places = {'P': CellType.PHALANX, 'S': CellType.STALK, 'T': CellType.TIP, 'SS': CellType.STALK}
    transition_list = ['time_S', 'time_SS', 'S-SS', 'SS-S', 'S-P', 'P-T', 'P-S']

    for place_name, place_type in places.items():
        net.add_place(Place(place_name, tokens=get_tokens(place_name, cell_matrix, vegf_matrix, cell_type=place_type)))
    # net.add_place(Place('SS', tokens=[]))

    # Add transition S time.
    net.add_transition(
        Transition('time_S', Expression(f"t[4] < {params['S_delay']}"))
    )
    net.add_input('S', 'time_S', Variable('t'))
    net.add_output('S', 'time_S', Expression('(t[0],t[1],t[2],t[3],t[4]+1)'))

    # Add transition SS time.
    net.add_transition(
        Transition('time_SS', Expression(f"t[4] < {params['SS_delay']}"))
    )
    net.add_input('SS', 'time_SS', Variable('t'))
    net.add_output('SS', 'time_SS', Expression('(t[0],t[1],t[2],t[3],t[4]+1)'))

    # Add transition P -> T.
    net.add_transition(
        Transition('P-T', Expression(f"t[2] > {params['d_min']} and t[3] > {params['alpha_V']}"))
    )
    net.add_input('P', 'P-T', Variable('t'))
    net.add_output('T', 'P-T', Variable('t'))

    # Add transition S -> SS.
    net.add_transition(Transition('S-SS', Expression(f"t[4] == {params['S_delay']}")))
    net.add_input('S', 'S-SS', Variable('t'))
    net.add_output('SS', 'S-SS', MultiArc([Expression('(t[0],t[1],t[2],t[3],0)'), Expression('(t[0]-1,t[1],t[2],t[3],0)')]))

    # Add transition SS -> S.
    net.add_transition(Transition('SS-S', Expression(f"t[4] == {params['SS_delay']}")))
    net.add_input('SS', 'SS-S', Variable('t'))
    net.add_output('S', 'SS-S', Variable('t'))

    # Add transition S -> P.
    net.add_transition(
        Transition('S-P', Expression(f"t[2] > {params['d_SP']}"))
    )
    net.add_input('S', 'S-P', Variable('t'))
    net.add_output('P', 'S-P', Variable('t'))

    # Add transition P -> S.
    net.add_transition(
        Transition('P-S', Expression(f"t[2] < {params['d_PS']}"))
    )
    net.add_input('P', 'P-S', Variable('t'))
    net.add_output('S', 'P-S', Variable('t'))

    return net, transition_list, places
