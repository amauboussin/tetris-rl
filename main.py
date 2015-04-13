from task import TetrisTask
from agents import *
from features import get_features
import numpy as np

board_width = 8
agent = Random(board_width)
task = TetrisTask(agent, width = board_width, height = 22, get_features = get_features)

state_histories, action_histories, reward_histories = task.run_trials(5000)

# print state_histories
# print action_histories
# print reward_histories

scores = []
for game in reward_histories:
	scores.append(sum(game))

print np.mean(scores)