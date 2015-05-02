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
