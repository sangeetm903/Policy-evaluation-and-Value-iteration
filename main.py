# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:23:04 2022

---------------------------------------------------------
Implementations of Value Iteration and policy Evaluations
---------------------------------------------------------

@author: Sangeet M
"""




import numpy as np
class IIScMess:
    def __init__(self):
        self.demand_values = [100, 200, 300, 400, 500]
        self.demand_probs = [0.15, 0.05, 0.3, 0.25, 0.25]
        self.capacity = self.demand_values[-1]
        self.days = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Weekend']
        self.cost_price = 10
        self.selling_price = 12
        self.action_space = [0, 100, 200, 300, 400, 500]
        self.state_space = [('Monday', 0)] + [(d, i)
                                              for d in self.days[1:] for i in [0, 100, 200, 300, 400]]

    def get_next_state_reward(self, state, action, demand):
        day, inventory = state
        result = {}
        result['next_day'] = self.days[self.days.index(day) + 1]
        result['starting_inventory'] = min(self.capacity, inventory + action)
        result['cost'] = self.cost_price * action
        result['sales'] = min(result['starting_inventory'], demand)
        result['revenue'] = self.selling_price * result['sales']
        result['next_inventory'] = result['starting_inventory'] - result['sales']
        result['reward'] = result['revenue'] - result['cost']
        return result

    def get_transition_prob(self, state, action):
        next_s_r_prob = {}
        for ix, demand in enumerate(self.demand_values):
            result = self.get_next_state_reward(state, action, demand)
            next_s = (result['next_day'], result['next_inventory'])
            reward = result['reward']
            prob = self.demand_probs[ix]
            if (next_s, reward) not in next_s_r_prob:
                next_s_r_prob[next_s, reward] = prob
            else:
                next_s_r_prob[next_s, reward] += prob
        return next_s_r_prob

    def is_terminal(self, state):
        day, inventory = state
        if day == "Weekend":
            return True
        else:
            return False


class IIScMessSolution:

    def example_policy(self, states):
        policy = {}
        for s in states:
            day, inventory = s
            prob_a = {}
            if inventory >= 200:
                prob_a[0] = 1
            else:
                prob_a[100 - inventory] = 0.4
                prob_a[300 - inventory] = 0.6
            policy[s] = prob_a
        return policy

    def iterative_policy_evaluation(self, env, policy, max_iter=1000, v=None, eps=0.01, gamma=1):
        if v is None:
            v=dict(zip(env.state_space, [0] * len(env.state_space)))
        for i in range(max_iter):
            diff=0
            states=env.state_space
            for s in states:# s state
                if env.is_terminal(s) is True:
                    continue
                policy_s=policy[s]
                sum_1=0#new policy value
                
                for a in policy_s.keys():# a action
                    tran_prob_sa=env.get_transition_prob(s,a)
                    tran_s1=tran_prob_sa.keys()
                    sum_2=0
                    for s_1 in tran_s1:
                        sum_2+=(tran_prob_sa[s_1]*gamma*(v[s_1[0]]+s_1[1]))
                    sum_1+=(policy_s[a]*sum_2)
                
                diff=max(diff,abs(sum_1-v[s]))
                v[s]=sum_1
                
            if eps>diff:
                break
        return v

    def value_iteration(self, env, max_iter=1000, eps=0.01, gamma=1):
        v=dict(zip(env.state_space, [0] * len(env.state_space)))
        for i in range(max_iter):
            diff=0
            states=env.state_space
            for s in states:
                if env.is_terminal(s) is True:
                    continue
                pol_set=[]
                actions=env.action_space
                for a in actions:
                    tran_prob_sa=env.get_transition_prob(s,a)
                    tran_s1=tran_prob_sa.keys()
                    sum_2=0
                    for s_1 in tran_s1:
                        sum_2+=(tran_prob_sa[s_1]*gamma*(v[s_1[0]]+s_1[1]))
                    pol_set.append(sum_2)
                max_pol=max(pol_set)
                diff=max(diff,abs(max_pol-v[s]))
                v[s]=max_pol
            if eps>diff:
                break

        return v


if __name__ == "__main__":
    # check the value_iteration output
    mess = IIScMess()
    solution = IIScMessSolution()
    v = solution.value_iteration(mess)
    assert (int(v[('Monday', 0)]) == 2884)

    # check the policy evaluation output
    mess = IIScMess()
    solution = IIScMessSolution()
    policy = solution.example_policy(mess.state_space)
    v = solution.iterative_policy_evaluation(mess, policy)
    assert (int(v[('Monday', 0)]) == 1775)


