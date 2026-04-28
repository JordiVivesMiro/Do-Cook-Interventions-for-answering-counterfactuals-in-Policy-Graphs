# Complementary materian of Interventionism In the Kitchen

This project helps generate and evaluate PGs and doPGs over the environment `overcooked-ai` v1.1.0 using the `pantheonrl` package. The policy graphs are extracted using `pgeon-xai` v1.0.1.

## Setup

Create a venv **specifically using** Python 3.8.

```shell
path/to/python3.8 -m venv venv
```

Using your venv, run `install.sh` from the project root. `pip` will warn that there are minor incompatibilites between some of the packages, but the project will run without problem.

```shell
source path/to/venv/source/activate
./install.sh
```

## Files

```
.
├── /agents    # Path to store saved agents
├── /experiments 
│   ├── /PGs       # Path where all PGs are stored
│   ├── /Trash     # Path where non-required generated files are sent
│   └── /doPGs
│       ├── /strat1 # Path were all doPGs generated following the first strategy are stored
│       ├── /strat2 # Path were all doPGs generated following the second strategy are stored
│       └── /strat3 # Path were all doPGs generated following the third strategy are stored
├── /IPGbaseCode   # Code required for executing base IPGs
├── /src           # Source code for pgeon-overcooked compatibility
│   ├── /discretizer            # Path with the required elements related to the discretizer
│   ├── doPolicyGraph.py        # doPG class
│   ├── interventional_PGT.py   # IIPG class
│   ├── interventionalNode.py   # Class corresponding to the nodes of the IIPG class
│   └── ...
├── install.sh 
├── README.md
├── requirements.txt
├── evaluateAgentsAndPgs.py     # Script to evaluate two elements corresponding to players 1 and 2 from a same layout (agents, PGs, doPGs)
├── generatePG.py               # Script to generate a PG from an agent
├── generateDoPG.py             # Script to generate a doPG from a base PG and an agent
├── generateAllDoPGs.py         # Script to generate all possible doPGs (following the 3 strategies and hiperparameters) from a base PG and agent
├── getDoPGmetrics.py           # Script to extract all 7 metrics from a doPG, the 5 population metrics will be shown in console and the two unitary ones saved as images
└── Questions.py                # Script to ask questions about an agent's behaviour using any PG or doPG
```

## Usage

All the .py scripts have command-line arguments to specify all variables of the execution.

```shell
python evaluateAgentsAndPgs.py    --layout LAYOUT
python generatePG.py --layout LAYOUT --episodes EPISODES --pov {player,partner}
python generateDoPG.py --layout LAYOUT --episodes EPISODES --pov {player,partner} --strategy {None,1,2,3} --k K --p p
python generateAllDoPGs.py --layout LAYOUT --episodes EPISODES --pov {player,partner}
python getDoPGmetrics.py --layout LAYOUT --pov {player,partner} --strategy {None,1,2,3} --k K --p p
python Questions.py --layout LAYOUT --pov {player,partner} --strategy {None,1,2,3} --k K --p p --question {why,do_over_another,dettect_inefficient} --node_idx NODE_IDX --action1 {UP,DOWN,RIGHT,LEFT,STAY,INTERACT} --action2 {UP,DOWN,RIGHT,LEFT,STAY,INTERACT}
```

Our previous work uses 3 layouts: `cramped_room`, `asymmetric_advantages`, `forced_coordination`.
