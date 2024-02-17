from mlagents_envs.environment import ActionTuple

def env_reset(env, behavior_name):
    env.reset()
    dec, term = env.get_steps(behavior_name)
    done = len(term.agent_id) > 0
    state = term.obs[0] if done else dec.obs[0]
    return state

def env_next_step(env, behavior_name, action):
    action_tuple = ActionTuple()
    action_tuple.add_continuous(action)
    env.set_actions(behavior_name, action_tuple)
    env.step()
    
    dec, term = env.get_steps(behavior_name)
    done = len(term.agent_id) > 0
    reward = term.reward if done else dec.reward
    next_state = term.obs[0] if done else dec.obs[0]
    
    return next_state, reward, done