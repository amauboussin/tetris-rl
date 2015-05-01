from agents import make_valid_actions, pieces, FittedQAgent
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import numpy as np
valid_actions = make_valid_actions(pieces)




class ParentAgent(FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {'n_estimators' : 50}):
        super(ParentAgent, self).__init__(N, self.get_actions, gamma, board_width, n_samples, regressor, regressor_params)
        self.child_agents = []

    def get_actions(self, state):
        return map (lambda x : x, range(len(self.child_agents)))

    def add_child(self, policy):
        self.child_agents.append(policy)

    def interact(self, state, reward, field, tet):
        self.print_reward = False
        if self.last_state != None and state != None:
            if not isinstance(self.last_action, list): self.last_action = [self.last_action]
            self.tuples.append((self.last_state, self.last_action, reward, state))
            self.n_tuples += 1
            if self.n_tuples == self.n_samples:
                self.print_reward = True
                self.regress()
                self.n_samples *= 2
        self.last_state = state

        #last action where action is the child agent
        self.last_action = self.current_policy(state, reward, field, tet)
        #action on the actual board
        self.board_action = self.child_agents[self.last_action](state, reward, field, tet)
        return self.board_action

    def get_policy(self):
        def make_list(a):
            if not isinstance(a, list): 
                a = [a]
            return a

        def learned_policy(state, reward, field, tet):
            if not state is None:
                next_sa = [(state, a) for a in self.get_actions(state) ]
                action = next_sa[np.argmax([self.reg.predict(sa[0]+ make_list(sa[1])) for sa in next_sa])][1]
                board_action = self.child_agents[action](state, reward, field, tet)
                return board_action

        return learned_policy


class LikesLeft (FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(LikesLeft, self).__init__(N, None, gamma, board_width, n_samples, regressor, regressor_params)

    def get_death_penalty(self):
        return 0

    def reward_function(self, state, next_state):
        return next_state[0] - state[0]

class LikesRight (FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(LikesRight, self).__init__(N, None, gamma, board_width, n_samples, regressor, regressor_params)

    def get_death_penalty(self):
        return 0

    def reward_function(self, state, next_state):
        return next_state[self.board_width-1] - state[self.board_width-1]

class LikesNoHoles (FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(LikesNoHoles, self).__init__(N, None, gamma, board_width, n_samples, regressor, regressor_params)

    def get_death_penalty(self):
        return 0

    def reward_function(self, state, next_state):
        return (next_state[self.board_width] - state[self.board_width])

class LikesFlat (FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(LikesFlat, self).__init__(N, None, gamma, board_width, n_samples, regressor, regressor_params)

    def get_death_penalty(self):
        return 0

    def reward_function(self, state, next_state):
        return sum(state[:self.board_width])