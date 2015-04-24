from agents import make_valid_actions, pieces, FittedQAgent
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
valid_actions = make_valid_actions(pieces)


class LikesLeft (FittedQAgent):

	def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
		super(LikesLeft, self).__init__(N, gamma, board_width, n_samples, regressor, regressor_params)

	def get_death_penalty(self):
		return 0

	def reward_function(self, state):
		return -1 * state[0]