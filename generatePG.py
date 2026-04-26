import argparse

import pgeon
import overcooked_ai_py.mdp.overcooked_mdp as mdp
import pantheonrl.common.agents            as prl
import stable_baselines3                   as sb3

from src.agent import OvercookedAgent
from src.discretizer.ego import DiscretizerOvercooked
from src.discretizer.predicates import Agent
from src.environment import OvercookedPantheonRLSingleAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str,
                        choices=('cramped_room',
                                 'asymmetric_advantages',
                                 'forced_coordination',),
                        default='cramped_room',
                        )
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--pov', type=str, choices=('player', 'partner'), default='player')
    args = parser.parse_args()

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

    # Create empty policy graph with an Overcooked discretizer
    discretizer = DiscretizerOvercooked(environment.env, Agent.PLAYER if args.pov == 'player' else Agent.PARTNER)
    pg = pgeon.PolicyGraph(environment, discretizer)

    # Generate policy graph
    pg = pg.fit(agentPov, num_episodes=args.episodes, update=False)

    # Save policy graph
    pg.save('csv', (f"./experiments/PGs/agent{1 if args.pov == 'player' else 2}_{args.layout}_nodes.csv",
                f"./experiments/PGs/agent{1 if args.pov == 'player' else 2}_{args.layout}_edges.csv",
                f"./experiments/Trash/agent{1 if args.pov == 'player' else 2}_{args.layout}_trajectories.csv"))
