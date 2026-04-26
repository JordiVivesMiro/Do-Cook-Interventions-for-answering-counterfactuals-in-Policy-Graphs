from gymnasium.wrappers import OrderEnforcing
from pgeon import Predicate

from src.discretizer.base import Discretizer
from src.discretizer.predicates import Held, PotState, Action2Nearest, Agent, Object, Direction


class DiscretizerOvercooked(Discretizer):
    def __init__(self,
                 env: OrderEnforcing,
                 pov: Agent
                 ):
        super().__init__(env,
                         pov,
                         predicate_space=(Held,
                                          Held,
                                          *(PotState for _ in env.mdp.get_pot_locations()),
                                          *([Action2Nearest] * 5),
                                          *(Action2Nearest for _ in env.mdp.get_pot_locations()),
                                          Direction))

    def discretize(self, state):
        state = state[1]

        # Temporary sources (coordination)
        temporary_sources = self._get_temporary_sources(state)

        has_2_pots = len(self.predicate_space) == 12

        return (Predicate(Held, [Agent.PLAYER, self._get_held_predicate(state, self.pov_agent_id)]),
                Predicate(Held, [Agent.PARTNER, self._get_held_predicate(state, 1 - self.pov_agent_id)]),
                *[Predicate(PotState, [pot, self._get_pot_state_predicate(state, pot)]) for pot in [Object.POT0] + ([Object.POT1] if has_2_pots else [])],
                Predicate(Action2Nearest, [Object.ONION, self._get_object_pos_predicate(obs=state, temporary_sources=temporary_sources, player_id=self.pov_agent_id, item='onion')]),
                Predicate(Action2Nearest, [Object.TOMATO, self._get_object_pos_predicate(obs=state, temporary_sources=temporary_sources, player_id=self.pov_agent_id, item='tomato')]),
                Predicate(Action2Nearest, [Object.SOUP, self._get_soup_pos_predicate(obs=state, temporary_sources=temporary_sources, player_id=self.pov_agent_id)]),
                Predicate(Action2Nearest, [Object.DISH, self._get_object_pos_predicate(obs=state, temporary_sources=temporary_sources, player_id=self.pov_agent_id, item='dish')]),
                Predicate(Action2Nearest, [Object.SERVICE, self._get_object_pos_predicate(obs=state, temporary_sources=temporary_sources, player_id=self.pov_agent_id, item='service')]),
                *[Predicate(Action2Nearest, [pot, self._get_pot_pos_predicate(obs=state, temporary_sources=temporary_sources, pot=pot, player_id=self.pov_agent_id)]) for pot in [Object.POT0] + ([Object.POT1] if has_2_pots else [])],
                Predicate(Direction, [self._get_partner_zone_predicate(state, self.pov_agent_id, 1 - self.pov_agent_id)])
                )

    def nearest_state(self, state):
        ...

    def str_to_state(self, state: str):
        predicates = state.split('+')
        if len(predicates) == 10:
            held_player, held_partner, pot, onion_act, tomato_act, soup_act, dish_act, service_act, pot_act, partner_pos = predicates
            held_player = Held[held_player[:-1].split('(')[1].split(';')[1]]
            held_partner = Held[held_partner[:-1].split('(')[1].split(';')[1]]
            pot = PotState[pot[:-1].split('(')[1].split(';')[1]]
            onion_act = Action2Nearest[onion_act[:-1].split('(')[1].split(';')[1]]
            tomato_act = Action2Nearest[tomato_act[:-1].split('(')[1].split(';')[1]]
            soup_act = Action2Nearest[soup_act[:-1].split('(')[1].split(';')[1]]
            dish_act = Action2Nearest[dish_act[:-1].split('(')[1].split(';')[1]]
            service_act = Action2Nearest[service_act[:-1].split('(')[1].split(';')[1]]
            pot_act = Action2Nearest[pot_act[:-1].split('(')[1].split(';')[1]]
            partner_pos = Direction[partner_pos[:-1].split('(')[1]]

            return (Predicate(Held, [Agent.PLAYER, held_player]),
                    Predicate(Held, [Agent.PARTNER, held_partner]),
                    Predicate(PotState, [Object.POT0, pot]),
                    Predicate(Action2Nearest, [Object.ONION, onion_act]),
                    Predicate(Action2Nearest, [Object.TOMATO, tomato_act]),
                    Predicate(Action2Nearest, [Object.SOUP, soup_act]),
                    Predicate(Action2Nearest, [Object.DISH, dish_act]),
                    Predicate(Action2Nearest, [Object.SERVICE, service_act]),
                    Predicate(Action2Nearest, [Object.POT0, pot_act]),
                    Predicate(Direction, [partner_pos])
                    )
        elif len(predicates) == 12:
            held_player, held_partner, pot0, pot1, onion_act, tomato_act, soup_act, dish_act, service_act, pot0_act, pot1_act, partner_pos = predicates
            held_player = Held[held_player[:-1].split('(')[1].split(';')[1]]
            held_partner = Held[held_partner[:-1].split('(')[1].split(';')[1]]
            pot0 = PotState[pot0[:-1].split('(')[1].split(';')[1]]
            pot1 = PotState[pot1[:-1].split('(')[1].split(';')[1]]
            onion_act = Action2Nearest[onion_act[:-1].split('(')[1].split(';')[1]]
            tomato_act = Action2Nearest[tomato_act[:-1].split('(')[1].split(';')[1]]
            soup_act = Action2Nearest[soup_act[:-1].split('(')[1].split(';')[1]]
            dish_act = Action2Nearest[dish_act[:-1].split('(')[1].split(';')[1]]
            service_act = Action2Nearest[service_act[:-1].split('(')[1].split(';')[1]]
            pot0_act = Action2Nearest[pot0_act[:-1].split('(')[1].split(';')[1]]
            pot1_act = Action2Nearest[pot1_act[:-1].split('(')[1].split(';')[1]]
            partner_pos = Direction[partner_pos[:-1].split('(')[1]]

            return (Predicate(Held, [Agent.PLAYER, held_player]),
                    Predicate(Held, [Agent.PARTNER, held_partner]),
                    Predicate(PotState, [Object.POT0, pot0]),
                    Predicate(PotState, [Object.POT1, pot1]),
                    Predicate(Action2Nearest, [Object.ONION, onion_act]),
                    Predicate(Action2Nearest, [Object.TOMATO, tomato_act]),
                    Predicate(Action2Nearest, [Object.SOUP, soup_act]),
                    Predicate(Action2Nearest, [Object.DISH, dish_act]),
                    Predicate(Action2Nearest, [Object.SERVICE, service_act]),
                    Predicate(Action2Nearest, [Object.POT0, pot0_act]),
                    Predicate(Action2Nearest, [Object.POT1, pot1_act]),
                    Predicate(Direction, [partner_pos])
                    )
        else:
            raise NotImplementedError

    def state_to_str(self, state) -> str:
        return '+'.join(str(pred) for pred in state)
