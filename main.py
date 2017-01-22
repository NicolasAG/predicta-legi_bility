# -*- coding: utf-8 -*-
from dataset_walker import dataset_walker as dw
import numpy as np
import cPickle
import argparse

#################
# COST FUNCTION #
#################

def cost(trajectory):
    """
    Define a cost function for any trajectory.
    :param trajectory: either a Call object or an itterable to apply the cost function on.
    :return: the length of that trajectory. We thus asume that shorter is better.
    """
    return len(trajectory)


#############################
#  DIFFERENT GOAL FUNCTIONS #
#############################

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

def goal_food_type(call):
    """
    Define a goal for any call to be a food type constraint.
    :param call: the Call object to extract the goal from.
    :return: a food type (string)
    """
    for constraint in call.labels['task-information']['goal']['constraints']:
        if "food" == constraint[0]:
            return constraint[1]
    return "dontcare"

def goal_price_range(call):
    """
    Define a goal for any call to be a price range constraint.
    :param call: the Call object to extract the goal from.
    :return: a price range (string)
    """
    for constraint in call.labels['task-information']['goal']['constraints']:
        if "pricerange" == constraint[0]:
            return constraint[1]
    return "dontcare"


def print_call(call):
    print "call.log[session-id]:", call.log["session-id"]
    print "cost:", call.cost
    print "predictability:", call.predictability
    print "legibility:", call.legibility
    for turn, label in call:
        print "bot:", turn["output"]["transcript"]
        print "user:", label["transcription"]
    print "call.labels[task-information]:", call.labels["task-information"]
    print "goal:", call.goal


def most_predictable_call(all_calls):
    """
    Get the most predictable call from a given list.
    :param all_calls: list of different calls.
    :return: the most predictable call for that list.
    :note: predictability of a Call object is already computed (when loaded).
    """
    call = all_calls[0]
    for c in all_calls[1:]:
        if c.predictability > call.predictability:
            call = c
    return call


def predictability(d):
    """
    Measure Predictability of a dialog (or a call). Something predictable is something with low cost.
    :param d: iterable to apply the cost function on (Call object, or array of utterances).
    :return: the predictability.
    :note: mostly for dialog = array of utterances since Call predictability is already computed.
    """
    return np.exp(-cost(d))

CACHE = {}  # map from starting bot utterance & goal to the most predictable following dialogue. Of the form:
# {
#     goal_1: {
#         <previous_bot_utterance - str> : <most predictable dialog - arr of utterances>,
#         ...
#     },
#     ...
# }

def most_predictable_dialog(all_calls, prev_utterance):
    """
    Get the most predictable dialogue for a given starting point from a list of calls.
    :param all_calls: list of different calls for the same goal.
    :param prev_utterance: dialogue must start by an utterance following this bot utterance.
    :return: the most predictable dialog for that starting point, and the number of dialogues with this constraint.
    """
    # Check the cache
    goal = all_calls[0].goal  # w.l.o.g. take the first call since they all have the same goal.
    if goal in CACHE and prev_utterance in CACHE[goal]:
        # print "  in cache!"
        return CACHE[goal][prev_utterance]

    # Look for it!
    dialogs = []  # list of possible dialogs; we take the most predictable at the end.
    num_calls = 0  # number of dialogs with the same starting point and same goal.
    for call in all_calls:
        good_call = False
        for turn, label in call:
            bot_u = turn["output"]["transcript"]
            user_u = label["transcription"]

            if good_call:  # if was previously flagged as good, append to the lastly created dialog.
                dialogs[-1].append(bot_u + " | " + user_u)

            # if good_call was reset to false and bot_u matches, flag as good and start a new dialog after that.
            elif not good_call and bot_u == prev_utterance:
                good_call = True
                dialogs.append([])  # create new empty dialog.
                num_calls += 1

    assert len(dialogs) == num_calls > 0
    # if len(dialogs) == 1:
    #     print "  nothing better than current call :("

    # Take the most predictable dialogue
    best_dialogue = dialogs[0]
    for d in dialogs[1:]:
        if predictability(d) > predictability(best_dialogue):
            best_dialogue = d

    # Save to CACHE:
    if goal not in CACHE:
        CACHE[goal] = {}
    CACHE[goal][prev_utterance] = best_dialogue

    return best_dialogue


def goal_proba(call, dialog, all_goals):
    """
    Compute the probability of this call's goal GIVEN that we already saw `dialog`.
    :param call: the call to compute the proba on.
    :param dialog: list of utterances seen so far.
    :param all_goals: map from all goals to their list of calls, and their most predictable call.
    :return: the proba of this goal given call up until time t.
    """
    init_pred = predictability(dialog)  # predictability of the dialog seen so far.

    bot_u = dialog[-1].split(" | ")[0]  # last bot utterance seen.
    calls = all_goals[call.goal][0]  # list of calls for that goal.
    best_remaining_pred = predictability(most_predictable_dialog(calls, bot_u))  # predictability of the most predictable remaining dialog.

    best_pred = all_goals[call.goal][1].predictability  # predictability of the most predictable call for that goal.

    num_of_calls = 0.
    for g, [c, _, _] in all_goals.iteritems():
        num_of_calls += len(c)
    goal_p = len(calls) / num_of_calls  # proba of that goal = #of calls with that goal / total #of calls.

    # print "    1)init_pred:", init_pred
    # print "    2)best_remaining_pred:", best_remaining_pred
    # print "    3)goal_p:", goal_p
    # print "    4)best_pred:", best_pred
    # print "    1*2*3/4:", init_pred * best_remaining_pred * goal_p / best_pred
    return init_pred * best_remaining_pred * goal_p / best_pred


def most_legible_call(all_calls):
    """
    Get the most legible call from a given list.
    :param all_calls: list of different calls.
    :return: the most legible call.
    """
    call = all_calls[0]
    for c in all_calls[1:]:
        if c.legibility > call.legibility:
            call = c
    return call


def main():

    parser = argparse.ArgumentParser(description='Measure Predictability and Legibility of dialogues.')
    parser.add_argument('goal_function', choices=['success','constraints','food', 'price'], help='goal function to use')
    args = parser.parse_args()

    gf = None
    if args.goal_function == "success":
        gf = goal_success
    elif args.goal_function == "constraints":
        gf = goal_constraints
    elif args.goal_function == "food":
        gf = goal_food_type
    elif args.goal_function == "price":
        gf = goal_price_range

    ###
    # Load data
    ###
    all_calls = dw(['dstc2_train', 'dstc2_dev', 'dstc2_test', 'dstc3_seed', 'dstc3_test'], cost, gf)
    print "all_calls:", len(all_calls)  # 5,510
    # assert len(all_calls) == len(dstc2_train) + len(dstc2_dev) + len(dstc2_test) + len(dstc3_seed) + len(dstc3_test)

    ###
    # Map from goals to their [[list of calls], most_predictable_call: None for now, most_legible_call: None for now]
    ###
    print "\nMap all different goals to their list of calls..."
    all_goals = {}
    for call in all_calls:
        if call.goal not in all_goals:
            all_goals[call.goal] = [[call], None, None]
        else:
            all_goals[call.goal][0].append(call)

    ###
    # Get list of calls from the `all_goals` dictionary to set their legibility.
    ###
    all_calls = []
    for g, [calls, _, _] in all_goals.iteritems():
        all_calls += calls
        print g, ":", len(calls)
    print "done: %d goals" % len(all_goals)

    ###
    # Get most predictable Call for each goal.
    ###
    for g, [calls, _, _] in all_goals.iteritems():
        print "\nLooking for most predictable call with goal", g, "from", len(calls), "calls."
        most_pred = most_predictable_call(calls)
        all_goals[g][1] = most_pred  # set the most predictable call for that goal.
        print_call(most_pred)

    ###
    # Measure legibility for each call.
    ###
    print "\n~~~~~~~~~~\nMeasuring Legibility for all calls..."
    for i, call in enumerate(all_calls):
        dialog = []  # observed dialog so far: list of alternating utterances.
        t = 0.  # current timestep
        total_p = 0.  # sum over all timestep t of P(G|s->t) * f(t)
        total_f = 0.  # sum over all timestep t of f(t)=len(call) - t

        def f(p):
            return len(call) - p

        for turn, label in call:
            # print "%d.%d / %d.%d" % (i, t, len(all_calls), len(call))
            bot_u = turn["output"]["transcript"]
            user_u = label["transcription"]

            dialog.append(bot_u + " | " + user_u)

            total_p += goal_proba(call, dialog, all_goals) * f(t)
            total_f += f(t)
            t += 1.

        assert t == len(call)
        call.legibility = total_p / total_f
    print "done.\n~~~~~~~~~~"

    ###
    # Get most legible Call for each goal.
    ###
    for g, [calls, _, _] in all_goals.iteritems():
        print "\nLooking for most legible call with goal", g, "from", len(calls), "calls."
        most_legi = most_legible_call(calls)
        all_goals[g][2] = most_legi   # set the most legible call for that goal.
        print_call(most_legi)

    print "\nSaving all goals with their list of calls, most predictable call, and most legible call..."
    f = open(
        "./calls_and_%s-goals.pkl" % args.goal_function,
        "wb"
    )
    cPickle.dump(all_goals, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print "done."


if __name__ == '__main__':
    main()

