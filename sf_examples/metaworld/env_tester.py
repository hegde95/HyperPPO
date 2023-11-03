from sf_examples.metaworld.train_metaworld import RandomizedMTEnv



render = True
render_mode = "human" if render else None

env = RandomizedMTEnv(eval = True, render_mode=render_mode)

for k in range(10):
    obs, info = env.reset()
    print(f"Resetting env to {env.env_name}...")

    for i in range(500):
        action = env.action_space.sample()
        if render:
            env.render()
        obs, reward, done, truncated, info = env.step(action)
