import random

import numpy as np
from sklearn.linear_model import SGDRegressor

# A = agent start, T = target, o = available path, # = wall, D = danger zone
# Required count: A=1, T=1, o=68, #=20, D=10
GRID = [
    "ooo#oooDoA",
    "o#ooo#oo#o",
    "o#D#ooDo#o",
    "oooo#ooooo",
    "#o#oD#o#oD",
    "ooDooooooo",
    "o#o##o#Doo",
    "ooooo#Doo#",
    "o#DoDooooo",
    "Tooooo#ooo",
]
START = (0, 9)
GOAL = (9, 0)

# Actions: up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

ROWS = len(GRID)
COLUMNS = len(GRID[0])
NUMBER_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_FEATURES = ROWS * COLUMNS * NUMBER_OF_ACTIONS
MAX_STEPS = 200

# Reward system (displayed on the Application page)
REWARDS = {
    "normal": -1,    # moving to a valid normal position
    "invalid": -5,   # attempting to move outside the grid
    "wall": -5,      # hitting a wall
    "danger": -10,   # entering a danger zone
    "goal": 50,      # reaching the goal
}


def environment_counts():
    # Count how many cells of each character the environment has.
    counts = {}
    for row in GRID:
        for char in row:
            counts[char] = counts.get(char, 0) + 1
    return counts


def step(state, action):
    # Returns next state, reward, whether the episode ended and the cell type.
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]

    # Reject moves outside the grid
    if not (0 <= row < ROWS and 0 <= column < COLUMNS):
        return state, REWARDS["invalid"], False, "Invalid"

    # Reject moves into a wall
    if GRID[row][column] == "#":
        return state, REWARDS["wall"], False, "Wall"

    next_state = (row, column)

    if next_state == GOAL:
        return next_state, REWARDS["goal"], True, "Goal"

    # Danger zones can be entered but they are penalized
    if GRID[row][column] == "D":
        return next_state, REWARDS["danger"], False, "Danger"

    return next_state, REWARDS["normal"], False, "Path"


def encode(state, action):
    # Encode one state-action pair as a one-hot vector.
    features = np.zeros(NUMBER_OF_FEATURES, dtype=float)
    state_index = state[0] * COLUMNS + state[1]
    feature_index = state_index * NUMBER_OF_ACTIONS + action
    features[feature_index] = 1.0
    return features


def predict_q_values(model, state):
    # Predict the Q-values of the 4 actions for a given state.
    features = np.array(
        [encode(state, action)
         for action in range(NUMBER_OF_ACTIONS)]
    )
    return model.predict(features)


def train(episodes=1000):

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    rng = random.Random(42)
    gamma = 0.95
    epsilon = 1.0
    epsilon_start = epsilon
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # Incremental linear model for Q-values.
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.1,
        random_state=42,
    )

    # Initialization
    model.partial_fit(
        np.zeros((1, NUMBER_OF_FEATURES)),
        np.array([0.0]),
    )
    successes = 0
    rewards = []

    # TRAINING
    for _ in range(episodes):

        state = START
        total_reward = 0

        for _ in range(MAX_STEPS):

            # Epsilon-greedy: explore or exploit
            if rng.random() < epsilon:
                action = rng.randrange(NUMBER_OF_ACTIONS)
            else:
                q_values = predict_q_values(model, state)

                best_actions = np.flatnonzero(
                    q_values == q_values.max()
                ).tolist()

                action = rng.choice(best_actions)

            next_state, reward, terminated, _ = step(state, action)

            # Q-learning target
            if terminated:
                target = float(reward)
            else:
                next_q_values = predict_q_values(model, next_state)
                target = reward + gamma * float(next_q_values.max())

            features = encode(state, action).reshape(1, -1)
            model.partial_fit(features, np.array([target]))

            state = next_state
            total_reward += reward

            if terminated:
                successes += 1
                break

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # EVALUATION (no exploration)
    state = START
    path = [state]
    steps = []
    evaluation_reward = 0
    visited = set()

    for number in range(1, MAX_STEPS + 1):
        q_values = predict_q_values(model, state)
        action = int(np.argmax(q_values))
        next_state, reward, terminated, cell_type = step(state, action)
        steps.append({
            "number": number,
            "state": state,
            "action": ACTION_NAMES[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward
        })

        evaluation_reward += reward
        path.append(next_state)

        # Stop if the policy falls into a loop
        if (state, action) in visited and not terminated:
            state = next_state
            break
        visited.add((state, action))

        state = next_state

        if terminated:
            break

    reached_goal = state == GOAL
    q_table = []

    for row in range(ROWS):
        for column in range(COLUMNS):
            position = (row, column)

            if GRID[row][column] != "#" and position != GOAL:

                q_table.append({
                    "state": position,
                    "action_values": [
                        round(value, 2)
                        for value in predict_q_values(
                            model, position
                        ).tolist()
                    ]
                })

    return {
        "episodes": episodes,
        "successes": successes,
        "success_percentage": round(successes / episodes * 100, 2),
        "average_reward": round(sum(rewards) / len(rewards), 2),
        "final_average": round(
            sum(rewards[-100:]) / len(rewards[-100:]), 2
        ),
        "final_epsilon": round(epsilon, 4),
        "reached_goal": reached_goal,
        "movements": len(steps),
        "evaluation_reward": evaluation_reward,
        "path": path,
        "steps": steps,
        "q_table": q_table,
        "parameters": {
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "max_steps": MAX_STEPS,
            "learning_rate": 0.1,
        },
        "rewards": REWARDS,
    }