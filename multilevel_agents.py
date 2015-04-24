from agents import make_valid_actions, pieces, FittedQAgent
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
valid_actions = make_valid_actions(pieces)


class LikesLeft (FittedQAgent):

	def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
		super(LikesLeft, self).__init__(N, gamma, board_width, n_samples, regressor, regressor_params)

	def get_death_penalty(self):
		return 0

	def reward_function(self, state):
		r = 0
		for i in range(self.board_width):
			r +=  -1 * (10-i)**2 * state[0]
		return r

class LikesRight (FittedQAgent):

	def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
		super(LikesRight, self).__init__(N, gamma, board_width, n_samples, regressor, regressor_params)

	def get_death_penalty(self):
		return 0

	def reward_function(self, state):
		r = 0
		for i in range(self.board_width):
			r += (10-i)**2 * state[0]
		return r