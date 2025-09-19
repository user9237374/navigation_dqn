from dqn_agent.agent import DqnAgent
from rainbow_dqn_agent.agent import RainbowDqnAgent
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for terminal
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
from tqdm import tqdm

def dqn(env, agent, agent_name="Agent", n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, checkpoint_name="checkpoint.pth"):
    """Deep Q-Learning.
    
    Params
    ======
        agent_name (str): Name of the agent for display
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        checkpoint_name (str): filename to save the trained weights
    """
    print(f"\n🌈 Training {agent_name}...")
    brain_name = env.brain_names[0]
    scores = []                        # list containing scores from each episode
    window_length = 100
    target_score = 13.0
    scores_window = deque(maxlen=window_length)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    # Create progress bar for the entire training
    pbar = tqdm(total=n_episodes, desc=f"Training {agent_name}", unit="episode")
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name] 
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        avg_score = np.mean(scores_window)
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'Avg Score': f'{avg_score:.2f}',
            'Current Score': f'{score:.1f}',
            'Epsilon': f'{eps:.3f}'
        })
        pbar.update(1)
        
        # Print milestone every 100 episodes
        if i_episode % window_length == 0:
            pbar.write(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}')
        if avg_score >= target_score:
            success_msg = f'\n🎉 {agent_name} solved environment in {i_episode} episodes! Average Score: {avg_score:.2f}'
            pbar.write(success_msg)
            pbar.close()
            torch.save(agent.network().state_dict(), checkpoint_name)
            pbar.write(f'💾 Weights saved as {checkpoint_name}')
            break
    
    pbar.close()
    return scores

def calculate_moving_average(scores, window=100):
        """Calculate moving average for scores."""
        moving_avg = []
        for i in range(len(scores)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(scores[start_idx:i+1]))
        return moving_avg

if __name__ == "__main__":

    # environment
    env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = env.brains[brain_name].vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    # device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting dual agent training comparison...")
    print(f"📊 Environment: {state_size} states, {action_size} actions")
    print(f"💻 Device: {device}")
    
    # Train agents
    dqn_agent = DqnAgent(state_size, action_size, device)
    rainbow_dqn_agent = RainbowDqnAgent(state_size, action_size, device)

    scores_dqn = dqn(env, 
                     dqn_agent, 
                     agent_name="DQN", 
                     n_episodes=1500,
                     eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                     checkpoint_name="dqn_checkpoint.pth")

    scores_rainbow = dqn(env, 
                         rainbow_dqn_agent, 
                         agent_name="Rainbow DQN", 
                         n_episodes=1500,
                         checkpoint_name="rainbow_dqn_checkpoint.pth")

    # Close environment
    env.close()
    
    # Calculate statistics
    dqn_final_avg = np.mean(scores_dqn[-100:]) if len(scores_dqn) >= 100 else np.mean(scores_dqn)
    rainbow_final_avg = np.mean(scores_rainbow[-100:]) if len(scores_rainbow) >= 100 else np.mean(scores_rainbow)
    
    # Print comparison summary
    print(f"\n🎯 Training Comparison Summary:")
    print(f"📈 DQN Agent:")
    print(f"   Episodes trained: {len(scores_dqn)}")
    print(f"   Final average score (last 100): {dqn_final_avg:.2f}")
    print(f"   Max score: {max(scores_dqn):.2f}")
    print(f"   Weights saved: dqn_checkpoint.pth")
    
    print(f"\n🌈 Rainbow DQN Agent:")
    print(f"   Episodes trained: {len(scores_rainbow)}")
    print(f"   Final average score (last 100): {rainbow_final_avg:.2f}")
    print(f"   Max score: {max(scores_rainbow):.2f}")
    print(f"   Weights saved: rainbow_dqn_checkpoint.pth")
    
    # Calculate moving averages
    dqn_moving_avg = calculate_moving_average(scores_dqn)
    rainbow_moving_avg = calculate_moving_average(scores_rainbow)
    
    # Plot 1: DQN raw scores and average
    plt.figure(figsize=(10, 6))
    plt.plot(scores_dqn, alpha=0.6, color='lightblue', label='DQN Raw Scores')
    plt.plot(dqn_moving_avg, color='blue', linewidth=2, label='DQN Average Score')
    plt.axhline(y=13.0, color='green', linestyle='--', alpha=0.7, label='Target (13.0)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training: Raw Scores vs Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dqn_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot 1 saved: dqn_training.png")
    
    # Plot 2: Rainbow DQN raw scores and average  
    plt.figure(figsize=(10, 6))
    plt.plot(scores_rainbow, alpha=0.6, color='pink', label='Rainbow Raw Scores')
    plt.plot(rainbow_moving_avg, color='red', linewidth=2, label='Rainbow Average Score')
    plt.axhline(y=13.0, color='green', linestyle='--', alpha=0.7, label='Target (13.0)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Rainbow DQN Training: Raw Scores vs Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rainbow_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot 2 saved: rainbow_training.png")
    
    # Plot 3: Compare average scores of both agents
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_moving_avg, color='blue', linewidth=2, label=f'DQN Average (Final: {dqn_final_avg:.2f})')
    plt.plot(rainbow_moving_avg, color='red', linewidth=2, label=f'Rainbow Average (Final: {rainbow_final_avg:.2f})')
    plt.axhline(y=13.0, color='green', linestyle='--', alpha=0.7, label='Target (13.0)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('DQN vs Rainbow DQN: Average Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("� Plot 3 saved: comparison.png")
    
    print(f"\n✅ Training complete! Check the saved files:")
    print(f"   🤖 DQN weights: dqn_checkpoint.pth")
    print(f"   🌈 Rainbow DQN weights: rainbow_dqn_checkpoint.pth")  
    print(f"   📊 Plots: dqn_training.png, rainbow_training.png, comparison.png")