import argparse
from IPGbaseCode.src.eval.desire_trials import get_desires
from src.interventional_PGT import InterventionalPolicyGraphAndTrajectories


def detect_inefficient_intentions(ipg: InterventionalPolicyGraphAndTrajectories, c_threshold: float=0.5, desires=None):
    """
    Detects inefficient intentions in the policy graph.
    """
    for node in ipg.nodes:
        ovservational_intention = []
        interventional_intention = []
        for action_idx in ['0', '1', '2', '3', '4', '5']:
            if node.node_id in node.transitions and action_idx in node.transitions[node.node_id]:
                ovservational_intention.append(node.answer_why(action_idx, c_threshold= 0))
            elif node.node_id in node.interventional_transitions and action_idx in node.interventional_transitions[node.node_id]:
                interventional_intention.append((action_idx, node.answer_why(action_idx, c_threshold= 0)))
        for desire in desires:
            maxObservational = 0
            maxInterventional = 0
            maxInterventional_action_idx = None
            desire_name = desire.name
            for elem in ovservational_intention:
                if desire_name in elem:
                    maxObservational = max(maxObservational, elem[desire_name]['expected'])
            for action_idx, elem in interventional_intention:
                if desire_name in elem:
                    if elem[desire_name]['expected'] > maxInterventional:
                        maxInterventional = elem[desire_name]['expected']
                        maxInterventional_action_idx = action_idx
            if maxObservational < maxInterventional:
                print(f"There is an inefficient behaviour in state {node.node_id} towards desire {desire.name}.")
                print(f"The highest observational intention was {maxObservational}")
                print(f"While the highest interventional intention {maxInterventional} for action {IDX_2_ACTION_MAP[maxInterventional_action_idx]}")
                print("---------------------------------------------------")

def why_action_over_another(iipg: InterventionalPolicyGraphAndTrajectories, node_idx: int, action_idx_1: str, action_idx_2: str, c_threshold: float=0.5, desires=None):
    node = iipg.nodes[node_idx]
    ints_1 = node.answer_why(action_idx_1, c_threshold= c_threshold, probability_threshold=-1)
    ints_2 = node.answer_why(action_idx_2, c_threshold= c_threshold, probability_threshold=-1)
    
    count1 = 0
    count2 = 0
    
    if ints_1 == {} and ints_2 == {}:
        print(f"Both actions {IDX_2_ACTION_MAP[action_idx_1]} and {IDX_2_ACTION_MAP[action_idx_2]} seem to not be intentional.")
        return
    else:
        for d in desires:
            if d.name not in ints_1 and d.name not in ints_2:
                continue
            print(f"Related to desire {d.name}:")
            if d.name in ints_1 and d.name not in ints_2:
                count1 += 1
                print(f"Action {IDX_2_ACTION_MAP[action_idx_1]} will contributes to the intention of the desire by", end=' ')
                if (ints_1[d.name]['expected'] > 0):
                    print(f"being expected to increase my intention by {ints_1[d.name]['expected']:.5f}. ", end=' ')
                elif (ints_1[d.name]['expected'] == 0):
                    print(f"being expected to keep my intention of fulfilling it the same. ", end=' ')
                else:
                    print(f"having a {ints_1[d.name]['prob_increase']:.2f} probability of an expected intention "
                        f"increase of {ints_1[d.name]['expected_pos_increase']:.5f}. ", end=' ')
                print(f"While action {IDX_2_ACTION_MAP[action_idx_2]} does not seem intentional over the desire.")
            elif d.name in ints_2 and d.name not in ints_1:
                count2 += 1
                print(f"Action {IDX_2_ACTION_MAP[action_idx_2]} contributes to the intention of the desire by", end=' ')
                if (ints_2[d.name]['expected'] > 0):
                    print(f"being expected to increase my intention by {ints_2[d.name]['expected']:.5f}. ", end=' ')
                elif (ints_2[d.name]['expected'] == 0):
                    print(f"being expected to keep my intention of fulfilling it the same. ", end=' ')
                else:
                    print(f"having a {ints_2[d.name]['prob_increase']:.2f} probability of an expected intention "
                        f"increase of {ints_2[d.name]['expected_pos_increase']:.5f}. ", end=' ')
                print(f"While action {IDX_2_ACTION_MAP[action_idx_1]} does not seem intentional over the desire.")
            elif d.name in ints_1 and d.name in ints_2:
                print(f"Both actions {IDX_2_ACTION_MAP[action_idx_1]} and {IDX_2_ACTION_MAP[action_idx_2]} have intention over the desire.")
                if ints_1[d.name]['expected'] > ints_2[d.name]['expected']:
                    count1 += 1
                    print(f"Action {IDX_2_ACTION_MAP[action_idx_1]} increases the intention by {ints_1[d.name]['expected'] - ints_2[d.name]['expected']} more than {IDX_2_ACTION_MAP[action_idx_2]}.")
                elif ints_1[d.name]['expected'] < ints_2[d.name]['expected']:
                    count2 += 1
                    print(f"Action {IDX_2_ACTION_MAP[action_idx_2]} increases the intention by {ints_2[d.name]['expected'] - ints_1[d.name]['expected']} more than {IDX_2_ACTION_MAP[action_idx_1]}.")
                else:
                    print(f"And both actions {IDX_2_ACTION_MAP[action_idx_1]} and {IDX_2_ACTION_MAP[action_idx_2]} provide the same expected intention increment towards the desire.")
        
def why(pg, node_idx, action_idx,  c_threshold):
    node = pg.nodes[node_idx]
    ints = node.answer_why(action_idx, c_threshold= c_threshold, probability_threshold=-1)
    # print("Intentions returned by why", ints)
    if ints == {}:
        print("Action seems to not be intentional.")
    for d, info in ints.items():
        if info['prob_increase'] > 0:
            print(f'I would do {IDX_2_ACTION_MAP[action_idx]} for the purpose of furthering {d} as', end=' ')
            if info['expected'] > 0:
                print(f"it is expected to increase my intention by {info['expected']:.2f}")
            elif info['expected'] == 0:
                print(f"it will keep my intention of fulfilling it the same.")
            else:
                print(f"it has a {info['prob_increase']:.2f} probability of an expected intention "
                    f"increase of {info['expected_pos_increase']:.2f}")   
        else:
            print(f"I would not do {IDX_2_ACTION_MAP[action_idx]} for the purpose of furthering {d} as", end=' ')
            print(f"it is expected to decrease my intention by {-info['expected']:.2f}")
                    
ACTION_2_IDX_MAP = {
    "up": "0",
    "down": "1",
    "right": "2",
    "left": "3",
    "stay": "4",
    "interact": "5"
}

# Inverse map: from action index string to action name
IDX_2_ACTION_MAP = {v: k for k, v in ACTION_2_IDX_MAP.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str,
                        choices=('cramped_room',
                                 'asymmetric_advantages',
                                 'forced_coordination',
                                 ),
                        default='cramped_room'
                        )
    parser.add_argument('--pov', type=str, choices=('player', 'partner'), default='player')
    parser.add_argument('--strategy', type=int, choices=(1, 2, 3), required=False,
                        help='Exploration strategy: 1, 2, or 3. If not set, use original PG.')
    parser.add_argument('--k', type=int, default=10, help='Parameter k for strategies 2 and 3')
    parser.add_argument('--p', type=int, default=30, help='Parameter p for strategy 3')
    parser.add_argument('--question', type=str, choices=('why', 'do_over_another', 'detect_inefficient'), default='why',
                        help='Which question to ask: why, do_over_another, or detect_inefficient')
    parser.add_argument('--node_idx', type=int, default=0, help='Node index for the question (default: 0)')
    parser.add_argument('--action1', type=str, choices=('up', 'down', 'left', 'right', 'stay', 'interact'), default='up',
                        help='First action name (default: up)')
    parser.add_argument('--action2', type=str, choices=('up', 'down', 'left', 'right', 'stay', 'interact'), default='up',
                        help='Second action name (default: up)')
    args = parser.parse_args()
    
    pov = args.pov
    layout = args.layout
    agent_num = 1 if pov == 'player' else 2

    # Select file paths based on strategy, matching getDoPGmetrics.py logic
    if args.strategy is None:
        nodes_path = f"./experiments/PGs/agent{agent_num}_{layout}_nodes.csv"
        edges_path = f"./experiments/PGs/agent{agent_num}_{layout}_edges.csv"
        interventional_edges_path = None
    elif args.strategy == 1:
        nodes_path = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_nodes.csv"
        edges_path = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_edges.csv"
        interventional_edges_path = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_interventional_edges.csv"
    elif args.strategy == 2:
        nodes_path = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_nodes.csv"
        edges_path = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_edges.csv"
        interventional_edges_path = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_interventional_edges.csv"
    elif args.strategy == 3:
        nodes_path = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_nodes.csv"
        edges_path = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_edges.csv"
        interventional_edges_path = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_interventional_edges.csv"
    else:
        raise ValueError("Invalid strategy")

    C_th = 0.5

    IPGT = InterventionalPolicyGraphAndTrajectories()
    IPGT.load_graph_from_files(
        edges_path,
        nodes_path,
        'Propositional',
        interventional_edges_path,
    )
    desires = get_desires(only_one_pot=layout == 'cramped_room')
    for desire in desires:
        IPGT.register_desire(desire)

    # Map action names to indices if needed
    action1 = ACTION_2_IDX_MAP[args.action1]
    action2 = ACTION_2_IDX_MAP[args.action2]

    if args.question == 'why':
        why(IPGT, node_idx=args.node_idx, action_idx=action1, c_threshold=C_th)
    elif args.question == 'do_over_another':
        why_action_over_another(IPGT, node_idx=args.node_idx, action_idx_1=action1, action_idx_2=action2, c_threshold=C_th, desires=desires)
    elif args.question == 'detect_inefficient':
        detect_inefficient_intentions(IPGT, c_threshold=C_th, desires=desires)
