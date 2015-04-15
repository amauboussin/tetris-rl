from task import TetrisTask
from fittedq import fittedQ
from agents import *
from features import get_features
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


def random_test(board_width = 8):
	agent = Random(board_width)
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)

	state_histories, action_histories, reward_histories = task.run_trials(1000)

	mean_score(reward_histories)

def fittedq_test(board_width = 8):
	agent = FittedQAgent()
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(100)
	mean_score(reward_histories)

def mean_score(reward_histories):
	scores = []
	for game in reward_histories:
		scores.append(sum(game))

	print np.mean(scores)

fittedq_test()
# random_test()