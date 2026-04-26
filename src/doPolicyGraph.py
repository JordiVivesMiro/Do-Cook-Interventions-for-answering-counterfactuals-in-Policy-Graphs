import gymnasium as gym
import csv

import tqdm
from typing import Optional, Tuple, Any, List
import random

from pgeon.policy_graph import PolicyGraph
from pgeon.agent import Agent
from pgeon.discretizer import Discretizer

class doPolicyGraph(PolicyGraph):
    def __init__(self,
                 environment: gym.Env,
                 discretizer: Discretizer
                 ):
        super().__init__(environment, discretizer)
        
        self.firstLevelStates = {}
        self._first_level_states_gathered = False
        self._interventional_edges = []
        self._int_edges_info = {}
        self._node_action_bloked = {}
        self._low_occupational_nodes = {}
    
    @staticmethod
    def from_nodes_and_edges(path_nodes: str,
                             path_edges: str,
                             path_interventional_edges: Optional[str],
                             environment: gym.Env,
                             discretizer: Discretizer):
        dopg = doPolicyGraph(environment, discretizer)

        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_edges_includes_csv = path_edges[-4:] == '.csv'
        if path_interventional_edges is not None:
            path_to_int_edges_includes_csv = path_interventional_edges[-4:] == '.csv'

        node_info = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for state_id, value, prob, freq in csv_r:
                state_prob = float(prob)
                state_freq = int(freq)

                node_info[int(state_id)] = {
                    'value': dopg.discretizer.str_to_state(value),
                    'probability': state_prob,
                    'frequency': state_freq
                }
                dopg.add_node(node_info[int(state_id)]['value'],
                            probability=state_prob,
                            frequency=state_freq)
        print(f"Loaded {len(node_info)} nodes from {path_nodes}")

        with open(f'{path_edges}{"" if path_to_edges_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for node_from, node_to, action, prob, freq in csv_r:
                node_from = int(node_from)
                node_to = int(node_to)
                # TODO Get discretizer to process the action id correctly;
                #  we cannot assume the action will always be an int
                action = int(action)
                prob = float(prob)
                freq = int(freq)

                dopg.add_edge(node_info[node_from]['value'], node_info[node_to]['value'], key=action,
                            frequency=freq, probability=prob, action=action)
          
        print(f"Loaded {len(dopg.edges)} edges from {path_edges}")
        
        if path_interventional_edges is not None:      
            with open(f'{path_interventional_edges}{"" if path_to_int_edges_includes_csv else ".csv"}', 'r+') as f:
                csv_r = csv.reader(f)
                next(csv_r)

                for node_from, node_to, action, prob, freq in csv_r:
                    node_from = int(node_from)
                    node_to = int(node_to)
                    # TODO Get discretizer to process the action id correctly;
                    #  we cannot assume the action will always be an int
                    action = int(action)
                    prob = float(prob)
                    freq = int(freq)

                    if (node_info[node_from]['value'], node_info[node_to]['value'], action) not in dopg._interventional_edges:
                        dopg._interventional_edges.append((node_info[node_from]['value'], node_info[node_to]['value'], action))

                    # Initialize nested dictionaries for from_node, to_node, and action if necessary
                    if node_info[node_from]['value'] not in dopg._int_edges_info:
                        dopg._int_edges_info[node_info[node_from]['value']] = {}
                    if node_info[node_to]['value'] not in dopg._int_edges_info[node_info[node_from]['value']]:
                        dopg._int_edges_info[node_info[node_from]['value']][node_info[node_to]['value']] = {}
                    if action not in dopg._int_edges_info[node_info[node_from]['value']][node_info[node_to]['value']]:
                        dopg._int_edges_info[node_info[node_from]['value']][node_info[node_to]['value']][action] = {'probability': prob, 'frequency': freq}
                
        print(f"Loaded {len(dopg._interventional_edges)} interventional edges from {path_interventional_edges if path_interventional_edges else 'None'}")
        dopg._is_fit = True
        return dopg
    
    def saveInterventionalNodesEdges(self, pathNodes: str, pathEdges: str, pathInterventionalEdges: str):
        print("Saving interventional nodes and edges to CSV files.")
        path_to_nodes_includes_csv = pathNodes[-4:] == '.csv'
        path_edges = pathEdges[-4:] == '.csv'
        path_to_edges_includes_csv = pathInterventionalEdges[-4:] == '.csv'

        # Collect all nodes from self.nodes and _interventional_edges

        # Create node_ids mapping
        node_ids = {}
        with open(f'{pathNodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['id', 'value', 'p(s)', 'frequency'])
            for elem_position, node in enumerate(self.nodes):
                node_ids[node] = elem_position
                csv_w.writerow([elem_position, self.discretizer.state_to_str(node), self.nodes[node]['probability'], self.nodes[node]['frequency']])

        with open(f'{pathEdges}{"" if path_edges else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['from', 'to', 'action', 'p(s)', 'frequency'])
            for edge in self.edges:
                state_from, state_to, action = edge
                csv_w.writerow([node_ids[state_from], node_ids[state_to], action,
                                self[state_from][state_to][action]['probability'],
                                self[state_from][state_to][action]['frequency']])

        # Save interventional edges
        with open(f'{pathInterventionalEdges}{"" if path_to_edges_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['from', 'to', 'action', 'p(s)', 'frequency'])
            sorted_list = list(
                sorted(self._interventional_edges, key=lambda item: (node_ids[item[0]], item[2], node_ids[item[1]]))
            )
            for edge in sorted_list:
                state_from, state_to, action = edge
                csv_w.writerow([node_ids[state_from], node_ids[state_to], action,
                                self._int_edges_info[state_from][state_to][action]['probability'],
                                self._int_edges_info[state_from][state_to][action]['frequency']])

    def _update_with_new_interventional_edge(self, edge: Tuple[Any, Any, Any]):
        """
        This function updates the interventional edges of the graph with the new edge found.
        """
        from_node, to_node, action = edge

        # Add the edge to the interventional edges if it doesn't already exist
        if edge not in self._interventional_edges:
            self._interventional_edges.append(edge)

        # Initialize nested dictionaries for from_node, to_node, and action if necessary
        if from_node not in self._int_edges_info:
            self._int_edges_info[from_node] = {}
        if to_node not in self._int_edges_info[from_node]:
            self._int_edges_info[from_node][to_node] = {}
        if action not in self._int_edges_info[from_node][to_node]:
            self._int_edges_info[from_node][to_node][action] = {'probability': 0, 'frequency': 0}

        # Increment the frequency
        self._int_edges_info[from_node][to_node][action]['frequency'] += 1
        if ((from_node, action) not in self._node_action_bloked):
            self._node_action_bloked[(from_node, action)] = 0
        self._node_action_bloked[(from_node, action)] += 1
        return self
    
    
    def _update_interventional_edges_probabilities(self):
        """
        This function updates the probabilities of the interventional edges.
        """
        print("Updating interventional edges probabilities.")
        edges_out = {}
        for edge in self._interventional_edges:
            from_node, to_node, action = edge
            if from_node not in edges_out:
                edges_out[from_node] = {}
            if action not in edges_out[from_node]:
                edges_out[from_node][action] = 0
            edges_out[from_node][action] += self._int_edges_info[from_node][to_node][action]['frequency']
        
        for edge in self._interventional_edges:
            from_node, to_node, action = edge
            self._int_edges_info[from_node][to_node][action]['probability'] = self._int_edges_info[from_node][to_node][action]['frequency'] / edges_out[from_node][action]
        return self

############################################################################################################
#                           Exploration strategies for Interventional Policy Graph
#############################################################################################################

    def interventionalExplorationStrategy1(self, agent, num_episodes: int = 10):
        progress_bar = tqdm.tqdm(range(num_episodes))
        progress_bar.set_description('Exploring following the first strategy...')
        for ep in progress_bar:
            self._runInterventionalEpisodeStrategy1(agent, seed=ep)
        self._update_interventional_edges_probabilities()
        self._normalize()

        return self
    
    def _runInterventionalEpisodeStrategy1(self, agent: Agent, seed: Optional[int] = None):
        observation, _ = self.environment.reset(seed=seed)
        done = False

        while not done:
            
            discretized_obs = self.discretizer.discretize(observation)
            
            actions = self._getActionsStategy1(discretized_obs)

            if len(actions) == 0:
                action = agent.act(observation)
                intervened = False
            else:
                action = random.choice(actions)
                intervened = True
                                    
            observation, _, done, done2, _ = self.environment.step(action)
            done = done or done2
            
            #if the discretized observation is not in the graph, add it
            if not self.has_node(self.discretizer.discretize(observation)):
                self.add_node(self.discretizer.discretize(observation), probability=0, frequency=0)
            
            if intervened:
                self._update_with_new_interventional_edge((discretized_obs, self.discretizer.discretize(observation), action))
        return self 
    
    def _getActionsStategy1(self, state):
        """
        This function returns the actions that have not been taken in the state.
        """
        listOfActions = set(list(range(self.environment.action_space.n)))
        for stuff in self[state].values():
            for action in stuff.keys():
                listOfActions.discard(action)
        
        return list(listOfActions)
    
    
    def interventionalExplorationStrategy2(self, agent, k, num_episodes: int = 10):
        progress_bar = tqdm.tqdm(range(num_episodes))

        progress_bar.set_description('Exploring following the second strategy...')
        for ep in progress_bar:
            self._runInterventionalEpisodeStrategy2(agent, k, seed=ep)
        self._update_interventional_edges_probabilities()
        self._normalize()

        return self
    
    def _runInterventionalEpisodeStrategy2(self, agent: Agent, k, seed: Optional[int] = None):
        observation, _ = self.environment.reset(seed=seed)
        done = False

        while not done:
            
            discretized_obs = self.discretizer.discretize(observation)

            actions = self._getActions2_3(discretized_obs, k)

            if len(actions) == 0:
                action = agent.act(observation)
                intervened = False
            else:
                action = random.choice(actions)
                intervened = True
                                    
            observation, _, done, done2, _ = self.environment.step(action)
            done = done or done2
            
            #if the discretized observation is not in the graph, add it
            if not self.has_node(self.discretizer.discretize(observation)):
                self.add_node(self.discretizer.discretize(observation), probability=0, frequency=0)
            
            if intervened:
                self._update_with_new_interventional_edge((discretized_obs, self.discretizer.discretize(observation), action))
            else:
                self._updateWithNewObservationalEdge2((discretized_obs, self.discretizer.discretize(observation), action))
        return self 
    
    def _updateWithNewObservationalEdge2(self, edge: Tuple[Any, Any, Any]):
        state_from, state_to, action = edge
                
        if not self.has_edge(state_from, state_to, key=action):
                self.add_edge(state_from, state_to, key=action, frequency=0, action=action)
        self[state_from][state_to][action]['frequency'] += 1
  
    def _getActions2_3(self, state, k):
        """
        This function returns the actions that have not been taken in the state.
        """

        listOfActions = set(list(range(self.environment.action_space.n)))
        for stuff in self[state].values():
            for action in stuff.keys():
                listOfActions.discard(action)
        auxList = []
        for action in listOfActions:
            if (state, action) in self._node_action_bloked:
                if self._node_action_bloked[(state, action)] >= k:
                    auxList.append(action)
        for action in auxList:
            listOfActions.discard(action)
        
        return list(listOfActions)
    
    
    def interventionalExplorationStrategy3(self, agent, k, p, num_episodes: int = 10):
        progress_bar = tqdm.tqdm(range(num_episodes))
        for node in self.nodes:
            if self.nodes[node]['frequency'] < p:
                self._low_occupational_nodes[node] = self.nodes[node]['frequency']

        progress_bar.set_description('Exploring following the third strategy...')
        for ep in progress_bar:
            self._runInterventionalEpisodeStrategy3(agent, k, p, seed=ep)
        self._update_interventional_edges_probabilities()
        self._normalize()

        return self
    
    def _runInterventionalEpisodeStrategy3(self, agent: Agent, k, p, seed: Optional[int] = None):
        observation, _ = self.environment.reset(seed=seed)
        done = False

        while not done:
            
            discretized_obs = self.discretizer.discretize(observation)
            
            if(discretized_obs in self._low_occupational_nodes):
                action = agent.act(observation)
                intervened = False
            else:
                actions = self._getActions2_3(discretized_obs, k)

                if len(actions) == 0:
                    action = agent.act(observation)
                    intervened = False
                else:
                    action = random.choice(actions)
                    intervened = True
                                    
            observation, _, done, done2, _ = self.environment.step(action)
            done = done or done2
            
            #if the discretized observation is not in the graph, add it
            if not self.has_node(self.discretizer.discretize(observation)):
                self.add_node(self.discretizer.discretize(observation), probability=0, frequency=0)
                self._low_occupational_nodes[self.discretizer.discretize(observation)] = 0
            
            if intervened:
                self._update_with_new_interventional_edge((discretized_obs, self.discretizer.discretize(observation), action))
            else:
                self._updateWithNewObservationalEdge3((discretized_obs, self.discretizer.discretize(observation), action), p)
        return self
    
    def _updateWithNewObservationalEdge3(self, edge: Tuple[Any, Any, Any], p):
        state_from, state_to, action = edge
        if state_from in self._low_occupational_nodes:
            self._low_occupational_nodes[state_from] += 1
            if self._low_occupational_nodes[state_from] >= p:
                del self._low_occupational_nodes[state_from]
                
        if not self.has_edge(state_from, state_to, key=action):
                self.add_edge(state_from, state_to, key=action, frequency=0, action=action)
        self[state_from][state_to][action]['frequency'] += 1