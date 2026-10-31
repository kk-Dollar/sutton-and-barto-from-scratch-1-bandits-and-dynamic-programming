"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    # TODO: Build a k-armed bandit testbed with seeded normal true values.
    rng=np.random.RandomState(seed)
    true_action_value=rng.normal(mean,std,k)
    return true_action_value

# Step 2 - pull_arm
def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    # TODO: Return the arm's true value plus unit-normal noise from rng
    return true_values[action] + rng.normal(0,1)

# Step 3 - sample_average_update
def sample_average_update(q_values, action_counts, action, reward):
    # TODO: Update an action-value estimate incrementally from one new reward...
    new_action_counts=action_counts.copy()
    new_q_values=q_values.copy()
    new_action_counts[action]+=1
    new_q_values[action]= new_q_values[action]+ (1/new_action_counts[action])*(reward-new_q_values[action])
    return new_q_values,new_action_counts

# Step 4 - epsilon_greedy_action
def epsilon_greedy_action(q_values, epsilon, rng):
    # TODO: Choose an action epsilon-greedily from the current value estimates.
    random= rng.random()
    k=q_values.size
    
    if random<epsilon:
        action_index=rng.integers(0,k)
        return action_index

    else:
        return np.argmax(q_values)
    
    return -1

# Step 5 - run_bandit_episode (not yet solved)
# TODO: implement

# Step 6 - track_rewards_and_optimal_actions (not yet solved)
# TODO: implement

# Step 7 - average_bandit_curves (not yet solved)
# TODO: implement

# Step 8 - apply_random_walk_drift (not yet solved)
# TODO: implement

# Step 9 - constant_step_size_update (not yet solved)
# TODO: implement

# Step 10 - optimistic_initialization (not yet solved)
# TODO: implement

# Step 11 - ucb_action_select (not yet solved)
# TODO: implement

# Step 12 - gradient_bandit_update (not yet solved)
# TODO: implement

# Step 13 - bandit_parameter_study (not yet solved)
# TODO: implement

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

