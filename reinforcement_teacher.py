# reinforcement_teacher.py
# Reference exercise from class: Q-Learning with SGDRegressor on a 4x4 grid.

import random
import numpy as np
from sklearn.linear_model import SGDRegressor

# 0 = open path, 1 = wall.
GRID = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 1, 0],
]
START = (0, 0)
GOAL = (3, 3)

# Row and column changes for each action.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

ROWS = len(GRID)
COLUMNS = len(GRID[0])
NUMBER_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_FEATURES = ROWS * COLUMNS * NUMBER_OF_ACTIONS


def step(state, action):
    """Return next state, reward, and termination flag."""
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]

    # Reject moves outside the grid or into a wall.
    if not (0 <= row < ROWS and 0 <= column < COLUMNS):
        return state, -5, False
    if GRID[row][column] == 1:
        return state, -5, False

    next_state = (row, column)

    # Reaching the goal ends the episode.
    if next_state == GOAL:
        return next_state, 20, True

    return next_state, -1, False


def encode(state, action):
    """Encode one state-action pair as a one-hot vector."""
    features = np.zeros(NUMBER_OF_FEATURES, dtype=float)
    state_index = state[0] * COLUMNS + state[1]
    feature_index = state_index * NUMBER_OF_ACTIONS + action
    features[feature_index] = 1.0
    return features


def predict_q_values(model, state):
    """Predict the Q-value of each available action."""
    features = np.array([
        encode(state, action)
        for action in range(NUMBER_OF_ACTIONS)
    ])
    return model.predict(features)


def train(episodes=1000):
    """Learn Q-values, then evaluate the policy."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    rng = random.Random(42)
    gamma = 0.95
    epsilon = 1.0

    # Incremental linear model for Q-values.
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.1,
        random_state=42,
    )

    # Initialize before calling predict().
    model.partial_fit(
        np.zeros((1, NUMBER_OF_FEATURES)),
        np.array([0.0]),
    )

    successes = 0
    rewards = []

    for _ in range(episodes):
        state = START
        total = 0
        for _ in range(100):
            if rng.random() < epsilon:
                action = rng.randrange(NUMBER_OF_ACTIONS)
            else:
                q_values = predict_q_values(model, state)
                best_actions = np.flatnonzero(
                    q_values == q_values.max()
                ).tolist()
                action = rng.choice(best_actions)

            next_state, reward, terminated = step(state, action)

            # Terminal states have no future reward.
            if terminated:
                target = float(reward)
            else:
                next_q_values = predict_q_values(model, next_state)
                target = reward + gamma * float(next_q_values.max())

            # Learn from one observed transition.
            features = encode(state, action).reshape(1, -1)
            model.partial_fit(features, np.array([target]))

            state = next_state
            total += reward
            if terminated:
                successes += 1
                break

        # Record one total per episode.
        rewards.append(total)
        epsilon = max(0.05, epsilon * 0.995)

    # Evaluate without updating the model.
    state = START
    path = [state]
    steps = []
    for number in range(1, 101):
        q_values = predict_q_values(model, state)
        action = int(np.argmax(q_values))
        next_state, reward, terminated = step(state, action)
        steps.append({
            "number": number,
            "state": state,
            "action": ACTION_NAMES[action],
            "next_state": next_state,
            "reward": reward,
        })
        path.append(next_state)
        state = next_state
        if terminated:
            break

    reached_goal = state == GOAL

    # Build a display table from model predictions.
    q_table = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            position = (row, column)
            if GRID[row][column] == 0 and position != GOAL:
                q_table.append({
                    "state": position,
                    "action_values": predict_q_values(
                        model, position
                    ).tolist(),
                })

    return {
        "episodes": episodes,
        "successes": successes,
        "final_average": round(
            sum(rewards[-100:]) / len(rewards[-100:]), 2
        ),
        "reached_goal": reached_goal,
        "path": path,
        "steps": steps,
        "q_table": q_table,
        "grid": GRID,
        "start": START,
        "goal": GOAL,
        "actions": ACTION_NAMES,
        "final_epsilon": round(epsilon, 4),
        "total_reward": sum(s["reward"] for s in steps),
        "num_steps": len(steps),
    }


if __name__ == "__main__":
    result = train(episodes=1000)
    print(f"Episodes: {result['episodes']}")
    print(f"Successful episodes: {result['successes']}")
    print(f"Average reward (last 100): {result['final_average']}")
    print(f"Reached goal: {result['reached_goal']}")
    print(f"Steps in evaluation: {result['num_steps']}")
    print(f"Total reward: {result['total_reward']}")
    print(f"Final epsilon: {result['final_epsilon']}")