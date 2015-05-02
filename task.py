from t_interface import *
import numpy as np

def mean_score(reward_histories):
	scores = []
	for game in reward_histories:
		if len(game) > 0:
			scores.append(sum(game))
	print np.mean(scores), max(scores)

class TetrisTask:

	def __init__(self, agent, width = 8, height = 22, piece_generator = TetrisRandomGenerator(), feature_function = lambda x,y : x, include_tet = True, display_death = False, death_penalty = 0):

		self.agent = agent
		self.game =TetrisGameEngine(width = width, height = height)
		self.piece_generator = piece_generator
		self.display_death = display_death
		self.death_penalty = death_penalty

		self.include_tet = include_tet
		self.width = width
		self.height = height

		self.get_features = feature_function

	def run_trials(self, trials = 100):

		state_histories = [ [] for t in xrange(trials)]
		action_histories = [ [] for t in xrange(trials)]
		reward_histories = [ [] for t in xrange(trials)]

		
		for trial in range(trials):
			#reset board
			self.game.__init__(width = self.width, height = self.height)
			field = self.game.get_field_state()

			reward = 0
			last_field = None #snapshot of the board before game over
			while (field):

				self.game.spawn(self.piece_generator.next())

				field = self.game.get_field_state()
				if not field: break #game over

				tet = self.game.tet_state[0]

				state = self.get_features(field, tet, self.include_tet)

				action = self.agent.interact(state, reward, field, tet)
				new_x, new_r = action

				state_histories[trial].append(state)
				action_histories[trial].append(action)
				reward_histories[trial].append(reward)
				if self.agent.print_reward:
					mean_score(reward_histories)

				self.game.set_rotation(new_r)
				valid_move = self.game.set_x(new_x)
				
				if not valid_move: break #game over

				self.game.hard_drop()
				reward = self.game.clear_lines()

				last_field = field if field else last_field

				
			if self.display_death: self.game.display(last_field)
			self.agent.interact(self.get_features(last_field, tet, self.include_tet), self.death_penalty, last_field, tet)
		return state_histories, action_histories, reward_histories










