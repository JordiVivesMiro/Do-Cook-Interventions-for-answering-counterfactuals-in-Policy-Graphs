import csv
import random
from domain.desire import Desire
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.interventionalNode import InterventionalNode

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../IPGbaseCode/src/eval')))
from graph import PolicyGraphAndTrajectories



class InterventionalPolicyGraphAndTrajectories(PolicyGraphAndTrajectories):
    TRANSITION_NODE_FREQ = 4
    
    def __init__(self):
        super().__init__()
        self.interventional_transitions = dict()
    
    def __verify(self):
        for node in self.nodes:
            try:
                p_dist = self.transitions[node.node_id]
                total_proba = 0
                for action_trans in p_dist.values():
                    for pf in action_trans.values():
                        total_proba += pf['probability']
                if total_proba > 1+1e-5:
                    raise ValueError(f"Exceeding Probability at node {node.node_id}: {total_proba}")
            except KeyError:
                if self.verbose:
                    print(f'Warning: State {node.node_id} has no sampled successors in the edge file')
        pass
    
    def load_graph_from_files(self, transitions_file, state_file, state_type, interventional_transitions_file=None):
        """
        Load the graph from files, including interventional transitions if provided.
        """
        if state_type == 'Propositional':
            self._load_propo_states(state_file)
        self._load_transitions(transitions_file)
        if interventional_transitions_file:
            self._load_interventional_transitions(interventional_transitions_file)
        self.__verify()
        self._min_transition_prob = 0.

        for node in self.nodes:
            node.nodes = self.nodes
            node.transitions = self.transitions
            node.interventional_transitions = self.interventional_transitions
              
    def _load_transitions(self, transitions_file):
        with open(transitions_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            self.transitions = {}
  
            for row in reader:
                src = int(row[self.TRANSITION_NODE_SOURCE])
                act = str(row[self.TRANSITION_NODE_ACTION])
                dest = int(row[self.TRANSITION_NODE_DEST])
                prob = float(row[self.TRANSITION_NODE_PROB])
                freq = float(row[self.TRANSITION_NODE_FREQ])
                src_node = self.transitions.get(src, {})
                act_node = src_node.get(act, {})
                act_node[dest] = {'probability': prob, 'frequency': freq}
                src_node[act] = act_node    
                self.transitions[src] = src_node
                self.nodes[dest].coinciders.add(src)
    
    def _load_propo_states(self, state_file):
        with open(state_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            self.nodes = [InterventionalNode(int(n_id), float(p), state_rep, self.nodes, self.transitions, self.interventional_transitions, self) for n_id, state_rep, p, _ in reader]
    
    def _load_interventional_transitions(self, transitions_file):
        with open(transitions_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            self.interventional_transitions = {}

            for row in reader:
                src = int(row[self.TRANSITION_NODE_SOURCE])
                act = row[self.TRANSITION_NODE_ACTION]
                dest = int(row[self.TRANSITION_NODE_DEST])
                prob = float(row[self.TRANSITION_NODE_PROB])
                freq = float(row[self.TRANSITION_NODE_FREQ])
                src_node = self.interventional_transitions.get(src, {})
                act_node = src_node.get(act, {})
                act_node[dest] = {'probability': prob, 'frequency': freq}
                src_node[act] = act_node
                self.interventional_transitions[src] = src_node
                
    def register_desire(self, desire: Desire):
        for node in self.nodes:
            p = node.check_desire(desire.clause, desire.action_idx)
            if p is not None:
                node.propagate_intention(desire, p)
                # node.increase_desire(desire, p)
        
    def get_action(self, state):
        # Find all nodes that match the state (len(b) == 0)
        matching_nodes = []
        for node in self.nodes:
            _, b, _ = node.compute_differences_rep(state)
            if len(b) == 0:
                matching_nodes.append(node)
        # If none, fallback to nodes with len(b) == 1
        if matching_nodes == []:
            for node in self.nodes:
                _, b, _ = node.compute_differences_rep(state)
                if len(b) == 1:
                    matching_nodes.append(node)
        if matching_nodes:
            # Aggregate action probabilities from all matching nodes
            action_prob_sum = {}
            for node in matching_nodes:
                action_probabilities = node.get_action_probability()
                if action_probabilities:
                    for action, prob in action_probabilities.items():
                        action_prob_sum[action] = action_prob_sum.get(action, 0) + prob
            if action_prob_sum:
                # Normalize probabilities
                total = sum(action_prob_sum.values())
                actions = list(action_prob_sum.keys())
                probabilities = [action_prob_sum[a] / total for a in actions]                
                return random.choices(actions, weights=probabilities, k=1)[0]
        print("No matching nodes found for state:", state)
        return random.choice(['0', '1', '2', '3', '4', '5'])
    
    def detect_inefficient_intentions(self, c_threshold: float=0.5, desires=None):
        nodeActions = {}
        for node in self.nodes:
            ovservational_intention = []
            interventional_intention = []
            for action_idx in ['0', '1', '2', '3', '4', '5']:
                if node.node_id in node.transitions and action_idx in node.transitions[node.node_id]:
                    ovservational_intention.append(node.answer_why(action_idx, c_threshold= c_threshold))
                elif node.node_id in node.interventional_transitions and action_idx in node.interventional_transitions[node.node_id]:
                    interventional_intention.append((node.answer_why(action_idx, c_threshold= c_threshold), action_idx))
            for desire in desires:
                maxObservational = 0
                maxInterventional = 0
                maxInterventionalAction = None
                desire = desire.name
                for elem in ovservational_intention:
                    if desire in elem:
                        maxObservational = max(maxObservational, elem[desire]['expected'])
                for obj in interventional_intention:
                    elem, action = obj
                    if desire in elem:
                        if elem[desire]['expected'] > maxInterventional:
                            maxInterventional = elem[desire]['expected']
                            maxInterventionalAction = action
                if maxObservational < maxInterventional:
                    if maxObservational < maxInterventional:
                        actions = nodeActions.get(node.node_id, [])
                        actions.append(maxInterventionalAction)
                        nodeActions[node.node_id] = actions
        return nodeActions

    def correct_inefficient_behavior(self, nodeActions):
        """
        Corrects the inefficient behavior by updating the interventional transitions.
        """
        for node_id, actions in nodeActions.items():
            for action in actions:
                destinations = self.interventional_transitions.get(node_id, {}).get(action, {})
                for dest, prob in destinations.items():
                    destinations[dest] = prob / (2 * len(actions))
                # reduce to half all probabilities of transitions for the node
                for action2, originalDests in self.transitions.get(node_id, {}).items():
                    for dest, pf in originalDests.items():
                        self.transitions[node_id][action2][dest]['probability'] = pf['probability'] / 2
                        
                self.transitions[node_id][action] = {}
                for dest, pf in destinations.items():
                    self.transitions[node_id][action][dest]['probability'] = pf['probability']
                
            
                    
    def reassignAllInfoOnNodes(self):
        """
        Reassigns all transitions to the interventional transitions.
        """
        for node in self.nodes:
            node.transitions = self.transitions
            node.interventional_transitions = self.interventional_transitions
            node.nodes = self.nodes
            node.pg = self
        print("Reassigned all transitions and interventional transitions to nodes.")
        
    def cleanIntentions(self):
        """
        Cleans intentions of all nodes.
        """
        for node in self.nodes:
            node.intention = {d: 0 for d in node.intention.keys()}
        print("Cleaned intentions of all nodes.")
        
    def saveInterventionalNodesEdges(self, pathNodes: str, pathEdges: str, pathInterventionalEdges: str):
        print("Saving interventional nodes and edges to CSV files.")
        path_to_nodes_includes_csv = pathNodes[-4:] == '.csv'
        path_edges = pathEdges[-4:] == '.csv'
        path_to_edges_includes_csv = pathInterventionalEdges[-4:] == '.csv'

        # Collect all nodes from self.nodes and _interventional_edges

        # Create node_ids mapping
        with open(f'{pathNodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['id', 'value', 'p(s)', 'frequency'])
            for node in self.nodes:
                csv_w.writerow([node.node_id, node.state_rep, node.probability, node.probability])

        with open(f'{pathEdges}{"" if path_edges else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['from', 'to', 'action', 'p(s)', 'frequency'])
            for state_from in self.transitions.keys():
                for action in self.transitions[state_from].keys():
                    for state_to in self.transitions[state_from][action].keys():
                        csv_w.writerow([state_from, state_to, action,
                                        self.transitions[state_from][action][state_to]['probability'],
                                        self.transitions[state_from][action][state_to]['frequency']])

        # Save interventional edges
        with open(f'{pathInterventionalEdges}{"" if path_to_edges_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['from', 'to', 'action', 'p(s)', 'frequency'])
            for state_from in self.interventional_transitions.keys():
                for action in self.interventional_transitions[state_from].keys():
                    for state_to in self.interventional_transitions[state_from][action].keys():
                        csv_w.writerow([state_from, state_to, action,
                                        self.interventional_transitions[state_from][action][state_to]['probability'],
                                        self.interventional_transitions[state_from][action][state_to]['frequency']])
    
    def getPopulationMetrics(self, c_threshold: float=0.5):
        numNodesWithIntention = 0
        numActionsAll = 0
        numActionsAllProb = 0
        numActionsIntentional = 0
        arrayActionsProb = []
        numTotalNodes = len(self.nodes)
        for node in self.nodes:
            actionsSeen = set()
            if node.node_id in self.transitions:
                actionsSeen.update(set(self.transitions[node.node_id].keys()))
            if node.node_id in self.interventional_transitions:
                actionsSeen.update(set(self.interventional_transitions[node.node_id].keys()))
            numActionsAll += len(actionsSeen)
            numActionsAllProb += len(actionsSeen) * node.probability
            if any(node.intention[d] >= c_threshold for d in node.intention):
                numNodesWithIntention += 1
                numActionsIntentional += len(actionsSeen)
                arrayActionsProb.append((node.probability, len(actionsSeen)))
        print(f"number of nodes with intention -> {numNodesWithIntention} / {numTotalNodes} ({numNodesWithIntention / numTotalNodes * 100:.2f}%)")
        print(f"aa -> {numActionsAll / len(self.nodes):.2f}")
        if numNodesWithIntention == 0:
            print("aai -> No nodes with intention found.")
        else:
            print(f"aai -> {numActionsIntentional / numNodesWithIntention:.2f}")
        print(f"aap -> {numActionsAllProb:.2f}")
        if numNodesWithIntention == 0:
            print("aaip -> No nodes with intention found.")
        else:
            totalIntentionProbs = sum(prob for prob, _ in arrayActionsProb)
            arrayActionsProb = [(p / totalIntentionProbs, a) for p, a in arrayActionsProb if p > 0]
            print(f"aaip -> {sum(p * a for p, a in arrayActionsProb):.2f}")
        print("---------------------------------------------------")
    
    def getUnitaryMetrics(self, pathInverventionImages, pathObservationImages, c_threshold: float=0.5):
        arrayInterventionsProbability = []
        arrayObservationsProbability = []
        arrayInterventionsNoProbability = []
        arrayObservationsNoProbability = []
        for node in self.nodes: 
            interventions = 0
            observations = 0
            if node.node_id in self.transitions:
                for tomap in node.transitions[node.node_id].values():
                    for pf in tomap.values():
                        observations += pf['frequency']
            if node.node_id in self.interventional_transitions:
                for tomap in node.interventional_transitions[node.node_id].values():
                    for pf in tomap.values():
                        interventions += pf['frequency']
            if node.probability != 0:
                arrayInterventionsProbability.append(interventions)
                arrayObservationsProbability.append(observations)
            else:
                arrayInterventionsNoProbability.append(interventions)
                arrayObservationsNoProbability.append(observations)
        arrayInterventionsProbability.sort(reverse=True)
        arrayInterventionsNoProbability.sort(reverse=True)
        arrayInterventions = arrayInterventionsProbability + arrayInterventionsNoProbability

        arrayObservationsProbability.sort(reverse=True)
        arrayObservationsNoProbability.sort(reverse=True)
        arrayObservations = arrayObservationsProbability + arrayObservationsNoProbability

        # Set figure with fixed size and DPI
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        fig.patch.set_facecolor('white')
        
        x_obs = list(range(len(arrayObservations)))
        split_obs = len(arrayObservationsProbability) - 0.5

        ax.bar(x_obs, arrayObservations, color='red', alpha=0.5, label='Observations', 
               edgecolor='none', width=1.0)
        # remove horizontal padding so first/last bars touch the axes
        if len(x_obs) > 0:
            ax.set_xlim(-0.5, len(x_obs) - 0.5)
        
        ax.set_yscale('log')
        ax.set_xlabel('Sorted nodes', fontsize=26)
        ax.set_ylabel('Number of actions', fontsize=26)
        ax.set_title('Observational transitions per node', fontsize=28)
        ax.axvline(x=split_obs, color='black', linewidth=2, label='Original / Non-Original split')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.legend(fontsize=26)
        fig.tight_layout()
        fig.savefig(pathObservationImages, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # ---------- bar plot: INTERVENTIONS ----------
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        fig.patch.set_facecolor('white')
        
        x_int = list(range(len(arrayInterventions)))
        split_int = len(arrayInterventionsProbability) - 0.5

        ax.bar(x_int, arrayInterventions, color='blue', alpha=0.5, label='Interventions',
               edgecolor='none', width=1.0)
        # remove horizontal padding so first/last bars touch the axes
        if len(x_int) > 0:
            ax.set_xlim(-0.5, len(x_int) - 0.5)

        ax.set_yscale('log')
        ax.set_xlabel('Sorted nodes', fontsize=26)
        ax.set_ylabel('Number of actions', fontsize=26)
        ax.set_title('Interventional transitions per node', fontsize=28)
        ax.axvline(x=split_int, color='black', linewidth=2, label='Original / Non-original split')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.legend(fontsize=26)
        fig.tight_layout()
        fig.savefig(pathInverventionImages, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
    def getNodesWithIntention(self, c_threshold: float=0.5):
        nodesWithIntention = {}
        for node in self.nodes:
            if (any(node.intention[d] >= c_threshold for d in node.intention)) and node.node_id not in nodesWithIntention:
                nodesWithIntention[node.node_id] = node.intention
        return nodesWithIntention
