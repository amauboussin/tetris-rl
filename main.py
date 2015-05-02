from task import TetrisTask
from agents import *
from features import get_features, heights, height_diffs, top_four
import numpy as np
import pylab as plt


def random_test(board_width = 8):
    trials = 50000
    agent = Random(board_width)
    task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)

    state_histories, action_histories, reward_histories = task.run_trials(trials)

    mean_score(reward_histories)
    reward_histories_hist(reward_histories, policy = 'Random')
    plt.show()
    reward_histories_time(reward_histories, policy = 'Random')
    plt.show()

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

def test_death_penalty(board_width = 8):
    for penalty in [0, -500, -1000, -5000]:
        print 'running for ', penalty, ' penalty'
        agent = FittedQAgent()
        task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features, death_penalty = penalty)
        state_histories, action_histories, training_reward_histories = task.run_trials(3000)
        new_agent = PolicyAgent(agent.current_policy)
        new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
        state_histories, action_histories, final_reward_histories = new_task.run_trials(1000)
        mean_score(final_reward_histories)
        reward_histories_time(training_reward_histories, policy = '%s penalty' % penalty, add_title = False)
    plt.title('Cumulative Reward by Game (varying game over penalty)')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Game #')
    plt.legend()
    plt.show()

def test_state_spaces(board_width = 8):
    feature_functions = [heights, height_diffs, top_four]
    names = ['Heights', 'Height Diffs', 'Top Four']
    for name, f in zip(names, feature_functions):
        print 'running for funtion ', name
        agent = FittedQAgent()
        task = TetrisTask(agent, width = board_width, height = 22, feature_function = f)
        state_histories, action_histories, training_reward_histories = task.run_trials(3000)
        new_agent = PolicyAgent(agent.current_policy)
        new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
        state_histories, action_histories, final_reward_histories = new_task.run_trials(1000)
        mean_score(final_reward_histories)
        reward_histories_time(training_reward_histories, policy = name, add_title = False)
    plt.title('Cumulative Reward by Game (varying state representation)')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Game #')
    plt.legend()
    plt.show()

def multireg_plot(board_width = 8):
    agent = MultiRegressorFittedQAgent(n_samples = 5000)
    task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
    state_histories, action_histories, reward_histories = task.run_trials(5000)

    agent = FittedQAgent(n_samples = 5000)
    task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
    state_histories2, action_histories2, training_reward_histories2 = task.run_trials(5000)

    reward_histories_time(reward_histories, policy = 'Multiple Regressors' , add_title = False)
    reward_histories_time(reward_histories, policy = 'One Regressor', add_title = False)

    plt.title('Cumulative Reward by Game (one vs. multiple regressors)')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Game #')
    plt.legend()
    plt.show()


def fittedq_test(board_width = 8):
    agent = FittedQAgent()
    task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features)
    state_histories, action_histories, training_reward_histories = task.run_trials(50000)
    # mean_score(reward_histories)
    new_agent = PolicyAgent(agent.current_policy)
    new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features)
    state_histories, action_histories, final_reward_histories = new_task.run_trials(1000)
    mean_score(final_reward_histories)

    reward_histories_hist(final_reward_histories, policy = 'FittedQ')
    plt.show()
    reward_histories_time(training_reward_histories, policy = 'FittedQ')
    plt.show()

def multiregfittedq_test(board_width = 8):
    agent = MultiRegressorFittedQAgent()
    task = TetrisTask(agent, width = board_width, height = 22, feature_function = get_features, include_tet = False)
    state_histories, action_histories, reward_histories = task.run_trials(10000)
    # mean_score(reward_histories)
    new_agent = PolicyAgent(agent.current_policy)
    new_task = TetrisTask(new_agent, width = board_width, height = 22, feature_function = get_features, include_tet = False)
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

# mirrorfittedq_test()
# multiregfittedq_test()
# test_n_estimators()
multireg_plot()
# random_test()

