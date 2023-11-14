#
for k in range(n_episodes):
    done=False
    obs = beam_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = beam_env.step(action)

        if done:
            beam_env.render(reward)
            # time.sleep(0.5)
            break

    beam_env.epoch=0
    
    beam_env.state=-np.ones([beam_env.n_state])
    
    beam_env.x_measured=beam_env.state[0:beam_env.n_meas]
    
    beam_env.f_measured=beam_env.state[beam_env.n_meas:beam_env.n_state]
    
    done=False
    while done ==False:
        action = experiment_design_sampling(beam_env)
        obs, reward, done, info = beam_env.step(action)
        print(reward)
        if done:
            table[k,4]=reward
            print(beam_env.x_measured)
            print(beam_env.f_measured)
            break


