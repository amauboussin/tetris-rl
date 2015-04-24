from t_interface import *
import numpy as np

def mean_score(reward_histories):
	scores = []
	for game in reward_histories:
		if len(game) > 0:
			scores.append(sum(game))
	print np.mean(scores), max(scores)

class TetrisTask:

	def __init__(self, agent, width = 8, height = 22, piece_generator = TetrisRandomGenerator(), feature_function = lambda x,y : x, display_death = False):

		self.agent = agent


		self.game =TetrisGameEngine(width = width, height = height)
		self.piece_generator = piece_generator
		self.display_death = display_death
		self.lines_cleared = 0


		self.width = width
		self.height = height

		self.get_features = feature_function

		#check if the agent has a custom reward function that maps state to reward
		if hasattr(self.agent, 'reward_function'):
			self.reward_function = self.agent.reward_function
		else:
			self.reward_function = lambda s : self.lines_cleared

		if hasattr(self.agent, 'get_death_penalty'):
			self.death_penalty = self.agent.get_death_penalty()
		else:
			self.death_penalty = -10000


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
				if not field: 
					reward_histories[trial][-1] = self.death_penalty
					break #game over, last thing led to death

				tet = self.game.tet_state[0]

				state = self.get_features(field, tet)

				action = self.agent.interact(state, reward, field, tet)
				new_x, new_r = action

				self.game.set_rotation(new_r)
				valid_move = self.game.set_x(new_x)
				

				state_histories[trial].append(state)
				action_histories[trial].append(action)
				reward_histories[trial].append(reward)
				if self.agent.print_reward:
					mean_score(reward_histories)

				if not valid_move: 
					reward_histories[trial][-1] = self.death_penalty
					break #game over, last thing led to death
				

				self.game.hard_drop()
				state = self.get_features(field, tet)
				self.lines_cleared = self.game.clear_lines()
				reward = self.reward_function(state)


				last_field = field if field else last_field

				
			if self.display_death: self.game.display(last_field)
		return state_histories, action_histories, reward_histories










