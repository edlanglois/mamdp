# MAMDP Experiments
Code for the paper "How RL Agents Behave When Their Actions Are Modified"
by Eric Langlois and Tom Everitt (AAAI 2021).

## Install
```sh
pip install .
```
This installs the mamdp package along with several scripts prefixed by `mamdp-`.

## Running Experiments
The following commands will reproduce the results described in the paper.

Run results are stored in the `experiments/` directory and are re-used if
available. If changing any parameters other than `NUM_RUNS`, make
sure that `experiments/` does not contain past runs.


### Train and evaluate the Simulation-Oversight environment
```sh
make -j<NUM_CPU_CORES> simulation-oversight
```
Training curves are saved to `experiments/simulation-oversight/training.png`
and can be plotted manually with:
```sh
mamdp-plot-evaluations experiments/simulation-oversight/
```

### Summarize the policies
```sh
mamdp-summarize-policies experiments/simulation-oversight/*.policies.json
```

### Train and evaluate the Small Whisky-Gold environment
```sh
make -j<NUM_CPU_CORES> NUM_RUNS=10 whisky-gold-small
```
### Summarize the Small Whisky-Gold strategies
0 is the state index at the branch point between heading directly to the goal
through the whisky (right; action = 3) or going around (down; action = 2)
```sh
mamdp-summarize-policies experiments/whisky-gold-small/*.policies.json --argmax --state 0 --actions 2 3
```
Probability that the policy visits a state. 11 is the index of the whisky
```sh
mamdp-plot-policies experiments/whisky-gold-small/*.eval.json --state 9
```

### Train and evaluate the Off-Switch environment
Uses a fixed learning rate instead of `1/visit_count`.
```sh
make -j<NUM_CPU_CORES> NUM_RUNS=10 off-switch
```

### Summarize the Off-Switch strategies
11 is the state index at the branch point between
detouring to the disable button (down; action = 2) or
heading directly towards the goal (left; action = 1).
```sh
mamdp-summarize-policies experiments/off-switch/*.policies.json --argmax --state 11 --actions 1 2
```

Probability that the policy visits a state. 36 is the index of the off switch
button state.
```sh
mamdp-plot-policies experiments/off-switch/*.eval.json --state 36
```

## Development
### Editable Install
```sh
python setup.py develop [--user]
```
Re-run this command to refresh the version number (based on git tags).

### Testing
```sh
python -m pytest
```

### Versioning
Uses [Semantic Versioning](https://semver.org/).

Versions are set exclusively via git tags:
```sh
git -a v0.1.2 -m "Version 0.1.2"
```
