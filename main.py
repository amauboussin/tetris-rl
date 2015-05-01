from task import TetrisTask
from agents import *
from multilevel_agents import *
from features import get_features
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

def reward_histories_time(reward_histories, policy = 'Random', add_title = True):
    #by trial
    flat = [sum([max(0,r) for r in trial]) for trial in reward_histories]

    cumulative = []; c = 0
    for r in flat:
        c += r
        cumulative.append(c)
    plt.plot(range(len(cumulative)), cumulative, ls = '-', label = policy)
    if add_title:
        plt.title('%s Policy Cumulative Reward by Game (100 points per line, %s games)' % (policy, len(reward_histories)))
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Game #')

def test_n_estimators(board_width = 8):
    for estimators in [50, 100, 200]:
        print 'running for ', estimators, ' estimators'
        agent = FittedQAgent(regressor_params = {'n_estimators' : estimators})
        task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
        state_histories, action_histories, training_reward_histories = task.run_trials(3000)
        new_agent = PolicyAgent(agent.current_policy)
        new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
        state_histories, action_histories, final_reward_histories = new_task.run_trials(1000)
        mean_score(final_reward_histories)
        
        reward_histories_time(training_reward_histories, policy = '%s estimators' % estimators, add_title = False)
    plt.title('Cumulative Reward by Game (varying n_estimators)')
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
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(5000)
	# mean_score(reward_histories)
	new_agent = PolicyAgent(agent.current_policy)
	new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
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
# multiregfittedq_test()
# random_test()
# fittedq_test()
test_child('LikesNoHoles')
