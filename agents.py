import random
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import numpy as np

#action is two numbers, an x coordinate followed by a rotation 
pieces = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']


def state_to_tet(state):
    return pieces[list(state[-7:]).index(1)]

def get_valid_actions(tet):
    bounding_box_width = 3
    board_width = 8
    leftmost = board_width - bounding_box_width + 1
    # +1 to compensate for exclusive bound of range function

    #square
    if tet == 'O': 
        return [[c, 0] for c in range(-1, leftmost)]

    #line
    if tet == 'I': 
        return [[c, 0] for c in range(0, leftmost-1)] 
        + [[c, 1] for c in range(-2, leftmost)] 

    #left column free on rotation 1, right column free on rotation 3
    else:
        return [[c,r] for c in range(0, leftmost) for r in [0,2]]
        + [[c,1] for c in range(-1, leftmost)]
        + [[c,3] for c in range(0, leftmost+1)]



def make_valid_actions(pieces):
    valid_actions = {}
    for p in pieces:
        valid_actions[p] = get_valid_actions(p)
    return valid_actions

valid_actions = make_valid_actions(pieces)

class Random:
    #TODO implement valid_actions that returns the set of valid actions
    def  __init__(self, board_width, action_function = None):
        #parameters go here

        self.last_state = None
        self.board_width  = board_width
        self.print_reward = False

        self.action_function = action_function

                   

    def get_actions(self, state):
        if self.action_function is None: 
            return valid_actions[state_to_tet(state)]
        else:
            return self.action_function(state)

    def interact(self, state, reward, field, tet):

        self.last_state = state
        valid = self.get_actions(state)
        action = random.choice(valid)
        return action

# with mirrored pieces
mirrored_rots = [0,3,2,1]
bounding_box_width = 3
board_width = 8
rightmost = board_width - bounding_box_width
mirrored_x = {}
mirrored_pieces = {'L':'J','Z':'S'}
binaries = {}
for i in range(-1, rightmost+2):
    mirrored_x[i] = rightmost - i

def state_to_tet_mir(state):
    val_pieces = ['I', 'O', 'T', 'J', 'S']
    return val_pieces[list(state[-5:]).index(1)]

def m_a(action):
    x = action[0]
    r = action[1]
    return [mirrored_x[x], mirrored_rots[r]]


def m_s(state):
    return state[-7::-1] + state[-6:]

class MirrorFittedQAgent:
    # number of iterations to regress, discount, board_width, num_samples, ?regressor?
    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        self.N = N
        self.gamma = gamma
        self.n_samples = n_samples
        self.regressor = regressor
        self.regressor_params = regressor_params
        self.reg = self.regressor(**self.regressor_params)
        r = Random(board_width)
        self.random = r
        self.current_policy = r.interact
        self.print_reward = False

        self.pieces = ['I', 'O', 'T', 'J', 'S']
        self.last_state = None
        self.last_action = None
        self.last_tet = None
        self.tuples = []
        self.n_tuples = 0

    def interact(self, state, reward, field, tet):
        self.print_reward = False
        if self.last_state != None and state != None:
            if self.last_tet in self.pieces:
                self.tuples.append((self.last_state, self.last_action, reward, state))
            else:
                self.tuples.append((m_s(self.last_state), self.last_action, reward, m_s(state)))
            self.n_tuples += 1
            if self.n_tuples == self.n_samples:
                self.print_reward = True
                self.regress()
                self.n_samples *= 2
        self.last_state = state
        self.last_tet = tet
        if tet in self.pieces:
            self.last_action = self.current_policy(state, reward, field, tet)
            return self.last_action
        else:
            self.last_action = self.current_policy(m_s(state), reward, field, mirrored_pieces[tet])
            return m_a(self.last_action)


    def regress(self):
        print 'regressing on %s tuples' % len(self.tuples)


        data = np.array([s+a for (s, a, r, new_s) in self.tuples])
        rewards = np.array([r for (s, a, r, new_s) in self.tuples])
        targets = np.array([r for (s, a, r, new_s) in self.tuples])

        new_data = ([new_s for (s, a, r, new_s) in self.tuples])

        reg = self.reg
        for i in range(self.N):
            reg.fit(data, targets)

            for j in range(len(data)):
                state = new_data[j]
                next_sa = np.array([state + a for a in  valid_actions[state_to_tet_mir(state)]])
                targets[j] = rewards[j] + self.gamma * np.amax(reg.predict(next_sa))

        def learned_policy(state, reward, field, tet):
            if tet not in self.pieces:
                print "Error"
            if not state is None:

                #next_sa = [state+a for a in valid_actions[state_to_tet_mir(state)] ]
                next_sa = [(state, a) for a in valid_actions[state_to_tet_mir(state)] ]
                #action = next_sa[np.argmax([reg.predict(sa) for sa in next_sa])][-2:]
                action = next_sa[np.argmax([reg.predict(sa[0]+sa[1]) for sa in next_sa])][1]
                return action

        self.current_policy = learned_policy
        print 'done'

class FittedQAgent(object):
    # number of iterations to regress, discount, board_width, num_samples, ?regressor?
    def __init__(self, N = 30, action_function = None, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        self.board_width = board_width
        self.N = N
        self.gamma = gamma
        self.n_samples = n_samples
        self.regressor = regressor
        self.regressor_params = regressor_params
        self.reg = self.regressor(**self.regressor_params)
        r = Random(board_width, action_function = action_function)
        self.random = r
        self.current_policy = r.interact
        self.print_reward = False

        self.last_state = None
        self.last_action = None
        self.tuples = []
        self.n_tuples = 0

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
        self.last_action = self.current_policy(state, reward, field, tet)
        return self.last_action

    def get_actions(self, state):
        return valid_actions[state_to_tet(state)]

    def regress(self):
        print 'regressing on %s tuples' % len(self.tuples)


        data = np.array([s+a for (s, a, r, new_s) in self.tuples])
        rewards = np.array([r for (s, a, r, new_s) in self.tuples])
        targets = np.array([r for (s, a, r, new_s) in self.tuples])

        new_data = ([new_s for (s, a, r, new_s) in self.tuples])

        reg = self.reg
        for i in range(self.N):
            reg.fit(data, targets)

            for j in range(len(data)):
                state = new_data[j]
                next_sa = np.array([state + a for a in self.get_actions(state)])
                targets[j] = rewards[j] + self.gamma * np.amax(reg.predict(next_sa))

        def learned_policy(state, reward, field, tet):
            if not state is None:
                next_sa = [state+a for a in self.get_actions(state) ]
                action = next_sa[np.argmax([reg.predict(sa) for sa in next_sa])][-2:]
                return action

        self.current_policy = learned_policy
        #random.shuffle(self.tuples)
        #self.tuples = self.tuples[:(self.n_samples*4/5)]
        print 'done'

# using a different regressor for each of the tetrominoes
# tetrominoes no longer encoded in state

class MultiRegressorFittedQAgent:
    # number of iterations to regress, discount, board_width, num_samples, ?regressor?
    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 10000, regressor = ExtraTreesRegressor, regressor_params = {}):
        self.N = N
        self.gamma = gamma
        self.n_samples = n_samples
        self.regressor = regressor
        self.regressor_params = regressor_params
        self.regs = {}
        r = Random(board_width)
        self.random = r
        self.current_policy = r.interact
        self.print_reward = False

        self.last_state = None
        self.last_action = None
        self.last_tet = None
        self.tuples = {}
        self.counts = {}
        self.n_tuples = 0

        for i in pieces:
            self.regs[i] = self.regressor(**self.regressor_params)
            self.tuples[i] = []
            self.counts[i] = 0

    def interact(self, state, reward, field, tet):
        self.print_reward = False
        if self.last_state != None and state != None:
            self.tuples[self.last_tet].append((self.last_state, self.last_action, reward, state))
            self.counts[self.last_tet] += 1
            if self.counts[self.last_tet] == self.n_samples:
                self.print_reward = True
                self.regress(self.last_tet)
                self.counts[self.last_tet] = 0
        self.last_state = state
        self.last_action = self.current_policy(state, reward, field, tet)
        self.last_tet = tet
        return self.last_action

    def regress(self, tet):
        print 'regressing on %s tuples' % len(self.tuples[tet])


        #for tet in pieces:
        print 'regressing for ' + tet
        cur_data = self.tuples[tet]
        actions = valid_actions[tet]
        data = np.array([s+a for (s, a, r, new_s) in cur_data])
        rewards = np.array([r for (s, a, r, new_s) in cur_data])
        targets = np.array([r for (s, a, r, new_s) in cur_data])

        new_data = ([new_s for (s, a, r, new_s) in cur_data])


        reg = self.regs[tet]
        for i in range(self.N):
            reg.fit(data, targets)

            for j in range(len(data)):
                state = new_data[j]
                next_sa = np.array([state + a for a in actions])
                targets[j] = rewards[j] + self.gamma * np.amax(reg.predict(next_sa))

        def learned_policy(state, reward, field, tet):
            if not state is None:
                reg = self.regs[tet]
                actions = valid_actions[tet]
                next_sa = [state+a for a in actions]
                action = next_sa[np.argmax([reg.predict(sa) for sa in next_sa])][-2:]
                return action

        self.current_policy = learned_policy
        #random.shuffle(self.tuples)
        #self.tuples = self.tuples[:(self.n_samples*4/5)]
        print 'done'

class MirrorMultiRegressorFittedQAgent:
    # number of iterations to regress, discount, board_width, num_samples, ?regressor?
    def __init__(self, N = 30, gamma = .98, board_width = 8, n_samples = 1000, regressor = ExtraTreesRegressor, regressor_params = {}):
        self.N = N
        self.gamma = gamma
        self.n_samples = n_samples
        self.regressor = regressor
        self.regressor_params = regressor_params
        self.regs = {}
        r = Random(board_width)
        self.random = r
        self.current_policy = r.interact
        self.print_reward = False
        self.pieces = ['I', 'O', 'T', 'J', 'S']

        self.last_state = None
        self.last_action = None
        self.last_tet = None
        self.tuples = {}
        self.limits = {}
        self.counts = {}
        self.n_tuples = 0

        for i in self.pieces:
            self.regs[i] = self.regressor(**self.regressor_params)
            self.tuples[i] = []
            self.limits[i] = n_samples
            self.counts[i] = 0

    def interact(self, state, reward, field, tet):
        self.print_reward = False
        if self.last_state != None and state != None:
            if self.last_tet in self.pieces:
                last_tet = self.last_tet
                last_state = self.last_state
                last_action = self.last_action
                s = state
            else:
                last_tet = mirrored_pieces[self.last_tet]
                last_state = m_s(self.last_state)
                last_action = m_a(self.last_action)
                s = m_s(state)
            self.tuples[last_tet].append((last_state, last_action, reward, s))
            self.counts[last_tet] += 1

            if self.counts[last_tet] == self.limits[last_tet]:
                self.print_reward = True
                self.limits[last_tet] *= 2
                self.regress(last_tet)
                self.counts[last_tet] = 0

        self.last_state = state
        self.last_tet = tet
        self.last_action = self.current_policy(state, reward, field, tet)
        return self.last_action

    def regress(self, tet):
        print 'regressing on %s tuples' % len(self.tuples[tet])


        #for tet in pieces:
        print 'regressing for ' + tet
        cur_data = self.tuples[tet]
        actions = valid_actions[tet]
        data = np.array([s+a for (s, a, r, new_s) in cur_data])
        rewards = np.array([r for (s, a, r, new_s) in cur_data])
        targets = np.array([r for (s, a, r, new_s) in cur_data])

        new_data = ([new_s for (s, a, r, new_s) in cur_data])


        reg = self.regs[tet]
        for i in range(self.N):
            reg.fit(data, targets)

            for j in range(len(data)):
                state = new_data[j]
                next_sa = np.array([state + a for a in actions])
                targets[j] = rewards[j] + self.gamma * np.amax(reg.predict(next_sa))

        def learned_policy(state, reward, field, tet):
            if not state is None:
                if not tet in self.pieces:
                    tet = mirrored_pieces[tet]
                    state = m_s(state)
                reg = self.regs[tet]
                actions = valid_actions[tet]
                next_sa = [state+a for a in actions]
                action = next_sa[np.argmax([reg.predict(sa) for sa in next_sa])][-2:]
                if not tet in self.pieces:
                    action = m_a(action)
                return action

        self.current_policy = learned_policy
        #random.shuffle(self.tuples)
        #self.tuples = self.tuples[:(self.n_samples*4/5)]
        print 'done'

class PolicyAgent:
    def  __init__(self, policy):

        self.print_reward = False
        self.policy = policy

    def interact(self, state, reward, field, tet):
        #determine using policy
        return self.policy(state, reward, field, tet)
