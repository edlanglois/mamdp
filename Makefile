# Settings
# --------

# Number of steps in each training run
NUM_TRAINING_STEPS:=100_000_000

# Number of independent training runs with different seeds
NUM_RUNS:=10

# Stop episodes at this many steps.
# 100 is the value used in the AI Safety Gridworlds paper for
# the Whisky-Gold and Off-Switch environments.
MAX_EPISODE_STEPS:=100

TRAINING_DISCOUNT_FACTOR:=0.99

# Extra arguments to pass to mamdp-train
TRAINING_ARGS:=

RESULTS_DIR:=experiments

# --------

.PHONY: all clean train eval

# Don't delete intermediate files
.SECONDARY:

# Perform a second expansion of rules
# Allows automatic variables like $(@D) to be used in the prerequisites.
.SECONDEXPANSION:

AGENTS:=q-learning virtual-sarsa empirical-sarsa es
# AGENTS+=random
# AGENTS+=policy-gradient
TRAINING_SEEDS:=$(shell seq $(NUM_RUNS))

define ENV_EXPERIMENTS
.PHONY: $1 eval-$1 train-$1

$1: $(RESULTS_DIR)/$1/training.pdf $(RESULTS_DIR)/$1/training.png

_AGENT_BASENAMES:=$$(addprefix $$(RESULTS_DIR)/$1/,$$(AGENTS))
_EXPT_BASENAMES:=$$(foreach agent_base,$$(_AGENT_BASENAMES),\
	$$(addprefix $$(agent_base).,$$(TRAINING_SEEDS)))

_EVAL_RESULTS:=$$(addsuffix .eval.json,$$(_EXPT_BASENAMES))

$(RESULTS_DIR)/$1/training.pgf: $$(_EVAL_RESULTS)

$(RESULTS_DIR)/$1/training.pdf: $$(_EVAL_RESULTS)

$(RESULTS_DIR)/$1/training.png: $$(_EVAL_RESULTS)

eval-$1: $$(_EVAL_RESULTS)

train-$1: $$(addsuffix .policies.json,$$(_EXPT_BASENAMES))

train: train-$1
eval: eval-$1
all: $1
endef

$(eval $(call ENV_EXPERIMENTS,simulation-oversight))
$(eval $(call ENV_EXPERIMENTS,exp-inverting-bandit))
$(eval $(call ENV_EXPERIMENTS,linear-inverting-bandit))
$(eval $(call ENV_EXPERIMENTS,off-switch))
$(RESULTS_DIR)/off-switch/%.policies.json: TRAINING_ARGS+="--learning-rate=0.1"
$(eval $(call ENV_EXPERIMENTS,whisky-gold-small))
$(eval $(call ENV_EXPERIMENTS,whisky-gold))

# Policy path structure:
# $(RESULTS_DIR)/<env>/<agent>.<seed>.policies.json
$(RESULTS_DIR)/%.policies.json: | $$(@D)/.
	mamdp-train \
		--env "$(shell basename $$(dirname $@))" \
		--agent "$(shell basename $@ | cut -d. -f1)" \
		--seed "$(shell basename $@ | cut -d. -f2)" \
		--steps "$(NUM_TRAINING_STEPS)" \
		--discount-factor "$(TRAINING_DISCOUNT_FACTOR)" \
		--save-policy-log-steps 0.05 \
		--max-episode-steps "$(MAX_EPISODE_STEPS)" \
		$(TRAINING_ARGS) \
		--quiet \
		--output "$@"

$(RESULTS_DIR)/%.eval.json: $(RESULTS_DIR)/%.policies.json
	mamdp-eval "$<" "$@" \
		--policy-dtype float \
		--max-episode-steps $(MAX_EPISODE_STEPS) \
		--quiet

$(RESULTS_DIR)/%.pgf:
	mamdp-plot-evaluations $^ --output "$@"

$(RESULTS_DIR)/%.pdf:
	mamdp-plot-evaluations $^ --output "$@" --size 6.5 4

$(RESULTS_DIR)/%.png:
	mamdp-plot-evaluations $^ --output "$@" --size 6.5 4

$(RESULTS_DIR)/.:
	mkdir "$@"

$(RESULTS_DIR)/%/.:
	mkdir -p "$@"

clean:
	rm -rf "$(RESULTS_DIR)"
