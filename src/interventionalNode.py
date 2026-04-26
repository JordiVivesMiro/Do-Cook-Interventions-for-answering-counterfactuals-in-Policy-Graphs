from typing import Dict, List, Set

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../IPGbaseCode/src/domain')))
from desire import Desire
from node import PropoNode

class InterventionalNode(PropoNode):
    def __init__(self, node_id: int, prob: float, state_rep,
                 nodes: List, transitions: Dict, interventional_transitions, pg):  # TODO: state_rep set of enum
        super().__init__(node_id, prob, state_rep, nodes, transitions, pg)
        self.interventional_transitions = interventional_transitions

    def get_action_probability(self):
        try:
            p_dist = self.transitions[self.node_id]
            action_prob_distro = dict()
            for action_idx, dest_state_distr in p_dist.items():
                action_prob_distro[action_idx] = sum([pf['probability'] for pf in dest_state_distr.values()])
            return action_prob_distro
        except KeyError:
            return dict()
        
    def propagate_intention(self, desire: Desire, probability, stop_criterion=1e-4):
        desire_name = desire.name
        self.update_intention(desire_name, probability)
        for coincider_idx in self.coinciders:
            coincider = self.nodes[coincider_idx]

            if coincider.check_desire(desire.clause, desire.action_idx) is None:
                coincider_transitions = self.transitions[coincider_idx].values()
            else:
                # If coincider can fulfill desire themselves, do not propagate it through the action_idx branch
                coincider_transitions = [v for action_idx, v in self.transitions[coincider_idx].items()
                                         if action_idx != desire.action_idx]
            prob_of_transition = 0
            for action_transitions in coincider_transitions:
                prob_of_transition += action_transitions.get(self.node_id, {'probability': 0, 'frequency': 0})['probability']
            # self.transitions = {n_idx: {action1:{dest_node1: P(dest1, action1|n_idx), ...}

            new_coincider_intention_value = prob_of_transition * probability
            if new_coincider_intention_value >= stop_criterion:
                try:
                    coincider.propagate_intention(desire, new_coincider_intention_value)
                except RecursionError:
                    print("Maximum recursion reach, skipping branch with intention of", new_coincider_intention_value)
                
    def answer_why(self, action_idx: str, c_threshold, probability_threshold=0):
        # probability_threshold: minimum probability of intention increase by which we attribute the agent is trying to
        # further an intention. eg: if it has 5% prob of increasing a desire but 95% of decreasing it
        attr_ints = {d: I_d for d, I_d in self.intention.items() if I_d >= c_threshold}
        if len(attr_ints) == 0:
            return {}
        else:
            successors = self.get_successors(action_idx, attr_ints.keys())
            int_increase = {}
            for d, val in attr_ints.items():
                int_increase[d] = dict()
                int_increase[d]['expected'] = 0
                int_increase[d]['prob_increase'] = 0
                int_increase[d]['expected_pos_increase'] = 0
                for _, p, ints in successors:
                    int_increase[d]['expected'] += p * ints.get(d, 0)
                    int_increase[d]['prob_increase'] += p if ints.get(d, 0) >= val else 0
                    int_increase[d]['expected_pos_increase'] += p * ints.get(d, 0) if ints.get(d, 0) >= val else 0
                int_increase[d]['expected'] -= val
                int_increase[d]['expected_pos_increase'] = \
                    int_increase[d]['expected_pos_increase']/int_increase[d]['prob_increase'] \
                        if int_increase[d]['prob_increase'] > 0 else 0
                int_increase[d]['expected_pos_increase'] -= val
                if int_increase[d]['prob_increase'] <= probability_threshold or int_increase[d]['expected'] + val < c_threshold:
                    # Action detracts from intention. If threshold =0, it always detracts. Else: it has at least
                    # 1-threshold probability of decreasing intention.
                    del (int_increase[d])
            return int_increase
    
    def check_desire(self, desire_clause: Set, action_id: int):
        # Returns None if desire is not satisfied. Else, returns probability of fulfilling desire
        #   ie: executing the action when in Node
        desire_clause_satisfied = True
        for atom in desire_clause:
            desire_clause_satisfied = desire_clause_satisfied and self.atom_in_state(atom)
            if not desire_clause_satisfied:
                return None
        return self.get_action_probability().get(action_id, 0)
    
    def get_successors(self, action_idx: str, desires: str):
        """
        Get the successors of the node for a given action.
        """
        try: 
            successors = [(s, pf['probability'], self.get_intention(s, desires)) for s, pf in
                        self.transitions[self.node_id][action_idx].items()]
            p_a_s = 0 # Prob of a given s to scale later (as successors contain P(s',a|s) )
            for _,p,_ in successors:
                p_a_s +=p
            successors = [(s, p/p_a_s, i) for s, p, i in successors]
        except KeyError:
            # print(f"No observarional successors for node {self.node_id} and action {action_idx}")
            successors = []
        try:
            interventional_successors = [(s, pf['probability'], self.get_intention(s, desires)) for s, pf in
                        self.interventional_transitions[self.node_id][action_idx].items()]
        except KeyError:
            # print(f"No interventional successors for node {self.node_id} and action {action_idx}")
            interventional_successors = []
        return successors + interventional_successors

    def get_intention(self, node_index: int, desires: str):
        """
        Get the intention of the node.
        """
        try:
            return self.nodes[node_index].intention
        except IndexError:
            return {d: 0 for d in desires}
        
    def compute_differences_rep(self, other_rep):
        self_rep = self.state_rep
        shared = self_rep.intersection(other_rep)
        added = other_rep - shared
        removed = self_rep - shared
        return shared, added, removed