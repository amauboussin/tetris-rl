from task import TetrisTask
from agents import *
from multilevel_agents import *
from features import get_features, get_mirror_features
import numpy as np
import pylab as plt
import sys

def random_test(board_width = 8):
	trials = 10000
	agent = Random(board_width)
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)

	state_histories, action_histories, reward_histories = task.run_trials(trials)

	mean_score(reward_histories)
	reward_histories_hist(reward_histories)

def reward_histories_hist(reward_histories, policy = 'Random'):
    plt.hist([sum(g) for g in reward_histories], color = 'steelblue')
    plt.title('%s Policy Reward Histogram (100 points per line, %s games)' % (policy, len(reward_histories)))
    plt.ylabel('# of games with reward')
    plt.xlabel('Reward')

x = np.arange(1000)
def reward_histories_time(results, policy = 'Random', add_title = True):
    length = np.shape(results)[1]
    for i in range(1,length):
    	results[:,i] += results[:,i-1]

    mean = np.mean(results[:,x], axis=0)
    std = np.std(results[:,x], axis=0)

    plt.fill_between(x, mean-2*std, mean+2*std, color='#d0d0d0')
    plt.plot(x, mean, ls = '-', label = policy)
    if add_title:
        plt.title('%s Policy Cumulative Reward by Game (10000 points per line, %s games)' % (policy, len(reward_histories)))
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Game #')

N = 5
def test_double(board_width = 8):
    agents = [(FittedQAgent, get_features), (MirrorFittedQAgent, get_mirror_features), (DoubleFittedQAgent, get_features)]
    for agent_init, ff in agents:
        results = np.zeros((N, 1000))
        for i in range(N):
            agent = agent_init()
            print 'running for ', agent.__class__.__name__, i
            task = TetrisTask(agent, width = board_width, height = 22, feature_function = ff)
            state_histories, action_histories, training_reward_histories = task.run_trials(1000)
            new_agent = PolicyAgent(agent.current_policy)
            new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = ff)
            # state_histories, action_histories, final_reward_histories = new_task.run_trials(10)
            results[i,:] = [sum([max(0,r) for r in trial]) for trial in training_reward_histories]
            # mean_score(final_reward_histories)
        reward_histories_time(results, policy = '%s' % agent.__class__.__name__, add_title = False)
    plt.title('Cumulative Reward by Game')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Game #')
    plt.legend()
    plt.show()

def test_multilevel(board_width = 8):
	agent = LikesRight()
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(5000)
	new_agent = PolicyAgent(agent.current_policy)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features, display_death = True)
	state_histories, action_histories, reward_histories = new_task.run_trials(100)
	mean_score(reward_histories)

def train_policy(agent_class, trials = 1000, board_width = 8):
	trials = 1000
	samples = 10000

	agent = agent_class(n_samples = samples)
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(trials)
	return agent.current_policy

def test_parent(board_width = 8):
	trials = 1000
	samples = 10000


	#make get_actions always return a list.
	#make Parent Agent work with a list

	right_policy = train_policy(LikesRight, trials = trials)
	left_policy = train_policy(LikesLeft, trials = trials)

	agent = ParentAgent(n_samples = samples)
	print 'running child'
	agent.add_child(right_policy)
	agent.add_child(left_policy)
	print 'running parent'
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(trials)
	print 'running policy'
	new_agent = PolicyAgent(agent.get_policy())
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features, display_death = True)
	state_histories, action_histories, reward_histories = new_task.run_trials(100)
	mean_score(reward_histories)

def test_child(child_name, board_width = 8):
	child = getattr(sys.modules['multilevel_agents'], child_name)
	trials = 100000
	samples = 10000

	no_holes_policy = train_policy(child, trials = trials)
	new_agent = PolicyAgent(no_holes_policy)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = new_task.run_trials(1000)
	mean_score(reward_histories)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features, display_death = True)
	state_histories, action_histories, reward_histories = new_task.run_trials(100)
	mean_score(reward_histories)

def fittedq_test(board_width = 8):
	agent = FittedQAgent()
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(10000)
	# mean_score(reward_histories)
	new_agent = PolicyAgent(agent.current_policy)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = new_task.run_trials(1000)
	mean_score(reward_histories)

def multiregfittedq_test(board_width = 8):
	agent = MultiRegressorFittedQAgent()
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(10000)
	# mean_score(reward_histories)
	new_agent = PolicyAgent(agent.current_policy)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = new_task.run_trials(1000)
	mean_score(reward_histories)

def mirrorfittedq_test(board_width = 8):
	agent = MirrorFittedQAgent()
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_mirror_features)
	state_histories, action_histories, reward_histories = task.run_trials(5000)
	# mean_score(reward_histories)
	new_agent = PolicyAgent(agent.current_policy)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_mirror_features)
	state_histories, action_histories, reward_histories = new_task.run_trials(1000)
	mean_score(reward_histories)

def mean_score(reward_histories):
	scores = []
	for game in reward_histories:
		if len(game): scores.append(sum(game))
	print max(scores)
	print np.mean(scores)

# test_parent()
# fittedq_test()
# mirrorfittedq_test()
test_double()
# multiregfittedq_test()
# random_test()
# fittedq_test()
# test_child('LikesNoHoles')
