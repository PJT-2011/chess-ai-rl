env = DummyVecEnv([lambda: ChessEnv()])

model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=100000)  # Increase for stronger model
model.save("ppo_chess_gpu")