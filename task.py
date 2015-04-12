from t_interface import *

class TetrisTask:

	def __init__(self, agent, width = 8, height = 22, piece_generator = TetrisRandomGenerator() ):

		self.agent = agent
		self.game =TetrisGameEngine(width = width, height = height)
		self.piece_generator = piece_generator

		self.width = width
		self.height = height




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
				action = self.agent.interact(field, reward)
				new_x, new_r = action

				state_histories[trial].append(field)
				action_histories[trial].append(action)
				reward_histories[trial].append(reward)

				self.game.set_rotation(new_r)
				valid_move = self.game.set_x(new_x)
				
				if not valid_move: break #game over

				self.game.hard_drop()
				reward = self.game.clear_lines()

				last_field = field if field else last_field

				
			#self.game.display(last_field)
		return state_histories, action_histories, reward_histories










