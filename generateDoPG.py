import argparse

import overcooked_ai_py.mdp.overcooked_mdp as mdp
import pantheonrl.common.agents            as prl
import stable_baselines3                   as sb3

from src.doPolicyGraph import doPolicyGraph
from src.agent import OvercookedAgent
from src.discretizer.ego import DiscretizerOvercooked
from src.discretizer.predicates import Agent
from src.environment import OvercookedPantheonRLSingleAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str,
                        choices=('cramped_room',
                                 'asymmetric_advantages',
                                 'forced_coordination',
                                 ),
                        default='cramped_room'
                        )
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--pov', type=str, choices=('player', 'partner'), default='player')
    parser.add_argument('--strategy', type=int, choices=(1, 2, 3), required=True,
                        help='Exploration strategy: 1, 2, or 3')
    parser.add_argument('--k', type=int, default=10, help='Parameter k for strategies 2 and 3')
    parser.add_argument('--p', type=int, default=30, help='Parameter p for strategy 3')
    args = parser.parse_args()

    # Load the layout    
    grid = mdp.OvercookedGridworld.from_layout_name(args.layout)

    # Load agents
    if args.pov == 'player':
        agentPov = OvercookedAgent(f'agents/agent1_{args.layout}_1M.zip')
        agentPartner = prl.StaticPolicyAgent(sb3.PPO.load(f'agents/agent2_{args.layout}_1M.zip').policy)
    else:
        agentPov = OvercookedAgent(f'agents/agent2_{args.layout}_1M.zip')
        agentPartner = prl.StaticPolicyAgent(sb3.PPO.load(f'agents/agent1_{args.layout}_1M.zip').policy)

    # Instantiate environment
    environment = OvercookedPantheonRLSingleAgent(grid, agentPartner, args.pov)
    # Prepare the discretizer
    discretizer = DiscretizerOvercooked(environment.env, Agent.PLAYER if args.pov == 'player' else Agent.PARTNER)
    # Load base policy graph
    pg = doPolicyGraph.from_nodes_and_edges(
        f"./experiments/PGs/agent{1 if args.pov == 'player' else 2}_{args.layout}_nodes.csv",
        f"./experiments/PGs/agent{1 if args.pov == 'player' else 2}_{args.layout}_edges.csv",
        None,
        environment, discretizer
    )

    if args.strategy == 1:
        pg = pg.interventionalExplorationStrategy1(agentPov, num_episodes=args.episodes)
        pg.saveInterventionalNodesEdges(
            f"./experiments/doPGs/strat1/agent{1 if args.pov == 'player' else 2}_{args.layout}_nodes.csv", 
            f"./experiments/doPGs/strat1/agent{1 if args.pov == 'player' else 2}_{args.layout}_edges.csv", 
            f"./experiments/doPGs/strat1/agent{1 if args.pov == 'player' else 2}_{args.layout}_interventional_edges.csv"
        )
    elif args.strategy == 2:
        pg = pg.interventionalExplorationStrategy2(agentPov, args.k, num_episodes=args.episodes)
        pg.saveInterventionalNodesEdges(
            f"./experiments/doPGs/strat2/agent{1 if args.pov == 'player' else 2}_{args.layout}_k={args.k}_nodes.csv", 
            f"./experiments/doPGs/strat2/agent{1 if args.pov == 'player' else 2}_{args.layout}_k={args.k}_edges.csv", 
            f"./experiments/doPGs/strat2/agent{1 if args.pov == 'player' else 2}_{args.layout}_k={args.k}_interventional_edges.csv"
        )
    elif args.strategy == 3:
        pg = pg.interventionalExplorationStrategy3(agentPov, args.k, args.p, num_episodes=args.episodes)
        pg.saveInterventionalNodesEdges(
            f"./experiments/doPGs/strat3/agent{1 if args.pov == 'player' else 2}_{args.layout}_k={args.k}_p={args.p}_nodes.csv", 
            f"./experiments/doPGs/strat3/agent{1 if args.pov == 'player' else 2}_{args.layout}_k={args.k}_p={args.p}_edges.csv", 
            f"./experiments/doPGs/strat3/agent{1 if args.pov == 'player' else 2}_{args.layout}_k={args.k}_p={args.p}_interventional_edges.csv"
        )
