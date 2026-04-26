import argparse

from IPGbaseCode.src.eval.desire_trials import get_desires
from src.interventional_PGT import InterventionalPolicyGraphAndTrajectories


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
    args = parser.parse_args()

    pov = args.pov
    layout = args.layout

    agent_num = 1 if pov == 'player' else 2

    if args.strategy is None:
        nodes_path = f"./experiments/PGs/agent{agent_num}_{layout}_nodes.csv"
        edges_path = f"./experiments/PGs/agent{agent_num}_{layout}_edges.csv"
        interventional_edges_path = None
        unitary_interventions_img = f"./experiments/PGs/agent{agent_num}_{layout}_unitary_interventions.png"
        unitary_observations_img = f"./experiments/PGs/agent{agent_num}_{layout}_unitary_observations.png"
    elif args.strategy == 1:
        nodes_path = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_nodes.csv"
        edges_path = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_edges.csv"
        interventional_edges_path = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_interventional_edges.csv"
        unitary_interventions_img = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_unitary_interventions.png"
        unitary_observations_img = f"./experiments/doPGs/strat1/agent{agent_num}_{layout}_unitary_observations.png"
    elif args.strategy == 2:
        nodes_path = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_nodes.csv"
        edges_path = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_edges.csv"
        interventional_edges_path = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_interventional_edges.csv"
        unitary_interventions_img = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_unitary_interventions.png"
        unitary_observations_img = f"./experiments/doPGs/strat2/agent{agent_num}_{layout}_k={args.k}_unitary_observations.png"
    elif args.strategy == 3:
        nodes_path = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_nodes.csv"
        edges_path = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_edges.csv"
        interventional_edges_path = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_interventional_edges.csv"
        unitary_interventions_img = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_unitary_interventions.png"
        unitary_observations_img = f"./experiments/doPGs/strat3/agent{agent_num}_{layout}_k={args.k}_p={args.p}_unitary_observations.png"
    else:
        raise ValueError("Invalid strategy")

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
    IPGT.getPopulationMetrics()
    IPGT.getUnitaryMetrics(unitary_interventions_img, unitary_observations_img)




