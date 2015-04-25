from task import TetrisTask
from agents import *
from multilevel_agents import *
from features import get_features
import numpy as np
import pylab as plt


def random_test(board_width = 8):
	trials = 10000
	agent = Random(board_width)
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)

	state_histories, action_histories, reward_histories = task.run_trials(trials)

	mean_score(reward_histories)
	plt.hist([sum(g) for g in reward_histories], color = 'steelblue')
	plt.title('Random Policy Reward Histogram (100 points per line, %s games)' % trials)
	plt.ylabel('# of games with reward')
	plt.xlabel('Reward')
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
	agent = agent_class()
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
	state_histories, action_histories, reward_histories = task.run_trials(trials)
	return agent.current_policy

def test_parent(board_width = 8):


	#make get_actions always return a list.
	#make Parent Agent work with a list

	right_policy = train_policy(LikesRight)
	left_policy = train_policy(LikesLeft)

	agent = ParentAgent()
	agent.add_child(right_policy)
	agent.add_child(left_policy)
	task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)

	state_histories, action_histories, reward_histories = task.run_trials(1000)
	
	new_agent = PolicyAgent(agent.current_policy)
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
	agent = MirrorMultiRegressorFittedQAgent()
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

test_parent()
# mirrorfittedq_test()
# multiregfittedq_test()
# random_test()
