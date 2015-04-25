from agents import make_valid_actions, pieces, FittedQAgent
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
valid_actions = make_valid_actions(pieces)




class ParentAgent(FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(ParentAgent, self).__init__(N, self.get_actions, gamma, board_width, n_samples, regressor, regressor_params)
        self.child_agents = []

    def get_actions(self, state):
        return range(len(self.child_agents))

    def add_child(self, policy):
        self.child_agents.append(policy)

    def interact(self, state, reward, field, tet):
        self.print_reward = False
        if self.last_state != None and state != None:
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



class LikesLeft (FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(LikesLeft, self).__init__(N, None, gamma, board_width, n_samples, regressor, regressor_params)

    def get_death_penalty(self):
        return 0

    def reward_function(self, state):
        r = 0
        for i in range(self.board_width):
            r +=  -1 * (10-i)**2 * state[0]
        return r

class LikesRight (FittedQAgent):

    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        super(LikesRight, self).__init__(N, None, gamma, board_width, n_samples, regressor, regressor_params)

    def get_death_penalty(self):
        return 0

    def reward_function(self, state):
        r = 0
        for i in range(self.board_width):
            r += (10-i)**2 * state[0]
        return r