from dataset_walker import dataset_walker as dw
import numpy as np


def cost(trajectory):
    """
    Define a cost function for any trajectory.
    :param trajectory: either a Call object or an itterable to apply the cost function on.
    :return: the length of that trajectory. We thus asume that shorter is better.
    """
    return len(trajectory)


def goal_success(call):
    """
    Define a goal for any call to be if it succeed.
    :param call: the Call object to extract the goal from.
    :return: binary goal: 0 if failure, 1 if success.
    """
    return 1 if call.labels['task-information']['feedback']['success'] else 0


def goal_constraints(call):
    """
    Define a goal for any call to be a list of constraints.
    :param call: the Call object to extract the goal from.
    :return: tuple of constraints, each of the form (<slot>, <value>). Must be tuples to store in dictionary keys.
    """
    goal = []
    for constraint in call.labels['task-information']['goal']['constraints']:
       goal.append(tuple(constraint))
    return tuple(goal)


def print_call(call):
    print "call.log[session-id]:", call.log["session-id"]
    print "cost:", call.cost
    print "predictability:", call.predictability
    for turn, label in call:
        print "bot:", turn["output"]["transcript"]
        print "user:", label["transcription"]
    print "call.labels[task-information]:", call.labels["task-information"]
    print "goal:", call.goal


def most_predictable(all_calls, goal):
    """
    Get the most predictable call for a given goal.
    :param all_calls: list of different calls.
    :param goal: restriction on the different calls.
    :return: the most predictable call for that goal, and the number of calls with that goal.
    """
    call = None
    num_call = 0
    for c in all_calls:
        if c.goal == goal:
            num_call += 1
            if call == None or c.predictability > call.predictability: 
                call = c
    return call, num_call


def most_predictable(all_calls, goal, next_utterance):
    #TODO


def goal_proba(call, dialog, all_calls, all_goals):
    """
    Compute the probability of this call's goal GIVEN that we saw the first t utterances.
    :param call: the call to compute the proba on.
    :param dialog: utterances seen so far.
    :param all_calls: list of all calls.
    :param all_goals: dictionary of all goals and their counts.
    :return: the proba of this goal given call up until time t.
    """
    init_pred = np.exp(-cost(dialog))  # predictability of the dialog seen so far.
    next_utterance = #TODO
    best_remaining_pred = np.exp(-cost(most_predictable(all_calls, goal, next_utterance)))  # predictability of the most predictable remaining dialog for that goal.
    best_pred = all_goals[call.goal][1]  # predictability of the most predictable call for that goal.
    goal_p = all_goals[call.goal][0] / len(all_calls)  # proba of that goal = #of calls with that goal / #of calls.

    return init_pred * best_remaining_pred * goal_p / best_pred


def main():
    # dstc2_train = dw('dstc2_train', cost, goal_success)
    # print "dstc2_train:", len(dstc2_train)  # 1,612
    # dstc2_dev = dw('dstc2_dev', cost, goal_success)
    # print "dstc2_dev:", len(dstc2_dev)  # 506
    # dstc2_test = dw('dstc2_test', cost, goal_success)
    # print "dstc2_test:", len(dstc2_test)  # 1,117

    # dstc3_seed = dw('dstc3_seed', cost, goal_success)
    # print "dstc3_seed:", len(dstc3_seed)  # 11
    # dstc3_test = dw('dstc3_test', cost, goal_success)
    # print "dstc3_test:", len(dstc3_test)  # 2,264

    all_calls = dw(['dstc2_train', 'dstc2_dev', 'dstc2_test', 'dstc3_seed', 'dstc3_test'], cost, goal_success)
    print "all_calls:", len(all_calls)  # 5,510
    # assert len(all_calls) == len(dstc2_train) + len(dstc2_dev) + len(dstc2_test) + len(dstc3_seed) + len(dstc3_test)

    print "\nGet all different goals..."
    all_goals = {}  # set of different goals with their count
    for call in all_calls:
        if call.goal not in all_goals:
            all_goals[call.goal] = [1.]
        else:
            all_goals[call.goal][0] += 1.

    for g, [count] in all_goals.iteritems():
        print "\nLooking for most predictable call with goal", g, "from", count, "calls."
        most_pred, _ = most_predictable(all_calls, g)
        print_call(most_pred)

        all_goals[g].append(most_pred.predictability)  # append the best predictability for that goal.

    print "\nMeasuring Legibility..."
    for call in all_calls:
        dialog = []  # observed dialog so far: list of alternating utterances.
        t = 0  # current timestep
        total_p = 0  # sum over all timestep t of P(G|s->t) * f(t)
        total_f = 0  # sum over all timestep t of f(t)=len(call) - t

        def f(p):
            return len(call) - p

        for turn, label in call:
            bot_u = turn["output"]["transcript"]
            user_u = label["transcription"]

            dialog.append(bot_u + " | " + user_u)

            total_p += goal_proba(call, dialog, all_calls, all_goals) * f(t)
            total_f += f(t)
            t += 1

        assert t == len(call)
        print "total_p:", total_p
        print "total_f:", total_f
        call.legibility = total_p / total_f
        print "legibility:", call.legibility


if __name__ == '__main__':
    main()

