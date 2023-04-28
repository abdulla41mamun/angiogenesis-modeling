import random

import snakes.plugins

@snakes.plugins.plugin("snakes.nets")
def extend(module):

    class Transition(module.Transition):
        def __init__(self, name: str, guard=None, **args):
            """
            Parameters
            ----------
            name: str
                Name of the transition.
            guard: snakes.nets.Expression
                Expression which value has to be True, for the transition to be enabled.
             args: dict
                Additional arguments.

            """
            self.probability = args.pop("probability", None)  # default probability is None
            self.time = 0.0
            self.delay = args.pop("delay", 0.0)  # By default, we don't want to delay the transition.
            module.Transition.__init__(self, name, guard, **args)

        def enabled(self, binding, **args):
            """
            Check whether the transition is available or not.

            Parameters
            ----------
            binding: snakes.data.Substitution
                The mode under which the transition is fired.
            args: dict
                Additional parameters.

            """
            r = random.random()
            # If an additional parameters 'untimed' is received, the delay is disregarded and the transition is fired.
            if self.probability is not None and r > self.probability:
                return False
            elif (self.time < self.delay) and not (args.pop("untimed", False) or self.delay == 0.0):
                return False
            else:
                return module.Transition.enabled(self, binding)

        def fire(self, binding):
            """
            Fire the transition with `binding`.

            Parameters
            ----------
            binding: snakes.data.Substitution
                The mode under which the transition is fired.

            """
            for place, label in self.input():
                place.remove(label.flow(binding))
            for place, label in self.output():
                place.add(label.flow(binding))
            # self.time = 0.0

        def reset(self):
            self.time = 0.0

    class Place(module.Place):
        def __init__(self, name, tokens=[], check=None, **args):
            """
            Parameters
            ----------
            name: str
                Name of the place.
            tokens: list
                List of tokens initially in the place.
            check:  snakes.typing.Instance
                A constraint on the tokens allowed in the place.
            args: dict
                Additional parameters.

            """
            self.post, self.pre = {}, {}
            module.Place.__init__(self, name, tokens, check, **args)

        def update_tokens(self, tokens):
            module.Place.reset(self, tokens)

        def reset(self, tokens):
            """
            Resets the places to the initial state.
            """
            # Handles token related aspects.
            module.Place.reset(self, tokens)
            # Reset the time for all the time constrained transitions.
            for name in self.post:   # Disable transitions in the post set.
                transition = self.net.transition(name)
                transition.time = 0.0

        def empty(self):
            """
            Remove all the tokens.
            """
            module.Place.empty(self)
            for name in self.post:  # Disable transitions in the post set.
                self.net.transition(name).time = 0.0

    class PetriNet(module.PetriNet):
        def reset(self):
            """
            Resets the marking of each place, and consequently all the transitions clocks.
            """
            self.set_marking(self.get_marking())

        def maximal_step(self):
            """
            Computes the largest time step that the petri net can performed without crossing any time constraint.
            """
            # Step computes the remaining time until the earliest time transition can be fired.
            step = 0.0
            for transition in self.transition():
                # If the transition time is not set, the transition does not have time constraints.
                if transition.delay == 0.0:
                    continue
                # If the delay is not met yet, we retrieve the remaining time until the transition can be fired.
                if transition.time < transition.delay:
                    if step == 0.0:
                        step = transition.delay - transition.time
                    else:
                        step = min(step, transition.delay - transition.time)
            return step

        def time_step(self, step: int = None):
            """
            Performs a time step smaller or equal to the one specified by the parameter 'step'. If the parameter is
            omitted, 'step' is set as the largest time step that the petri net is allowed to performed without
            crossing any time constraint.

            Parameters
            ---------
            step: int
                Time step to be performed.

            """
            if step is None:    # Calculate the maximum step that can be taken.
                step = self.maximal_step()
            # If the specified step is greater than the time step allowed for not crossing any constraint,
            # the time step is shrank accordingly.
            else:
                step = min(self.maximal_step(), step)
            # Performing the time step for each time transition.
            for transition in self.transition():
                transition.time += step
            return step

        def unit_step(self):
            """
            Performs a one unit time step.
            """
            for transition in self.transition():
                transition.time += 1

    return Transition, Place, PetriNet
