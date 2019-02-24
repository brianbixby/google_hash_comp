import argparse
import os
from trainer import game
from trainer.policy_gradient import PolicyGradient
import numpy as np

rewards = []
RENDER_REWARD_MIN = 10

R = 6
C = 7
X_DIM = R * C * 2 + 5
ACTIONS = ["right", "down", "left", "up", "cut_right", "cut_left", "cut_up", "cut_down"]

def preprocess(state_dict):
	state = np.concatenate((
		np.array(state_dict['ingredients_map']).ravel(),
		np.array(state_dict['slices_map']).ravel(),
		np.array(state_dict['cursor_position']).ravel(),
		[state_dict['slice_mode'],
		state_dict['min_each_ingredient_per_slice'],
		state_dict['max_ingredients_per_slice']],
	))
	return state.astype(np.float).ravel()

def main(args):
    print('main')
    args_dict = vars(args)
    print('args: {}'.format(args_dict))

    PG = PolicyGradient(
            n_x = X_DIM,
            n_y = 8,
            learning_rate=args.learning_rate,
            reward_decay=args.reward_decay,
            output_dir=args.output_dir,
            max_to_keep=args.max_to_keep,
            restore=args.restore,
            save_checkpoint_steps=args.save_checkpoint_steps
            )

    for episode in range(args.episodes):
        env = game.Game({'max_steps':200})
        episode_reward = 0
        h = 5			
        l = 1
        pizza_lines = ["TMTMMTT","MMTMTMM", "MTTMTTT", "TMMMTMM", "TTMTTTM", "TMTMTMT"]
        pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
        state = env.init(pizza_config)[0]  #np.zeros(OBSERVATION_DIM) #get only first value of tuple
        while True:
            if args.render: 
                env.render()
            # sample one action with the given probability distribution
            # 1. Choose an action based on observation
            action = PG.choose_action(state)

            # 2. Take action in the environment
            state_, reward, done, info = env.step(ACTIONS[action])

            # 3. Store transition for training
            PG.store_transition(preprocess(state), action, reward)
            
            # Save new state
            #state = state_
            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)

                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                # Render env if we get to rewards minimum
                if max_reward_so_far > RENDER_REWARD_MIN: #args.render = True
                    break
                h = np.random.randint(1, R * C + 1)
                l = np.random.randint(1, h // 2 + 1)
                env = game.Game({'max_steps':2000}) # initialize game from game.py
                pizza_lines = ["TMMMTTT","MMMMTMM", "TTMTTMT", "TMMTMMM", "TTTTTTM", "TTTTTTM"]
                pizza_config = { 'pizza_lines': pizza_lines, 'r': R, 'c': C, 'l': l, 'h': h }
            # Save new state
            state = state_
        # if args.render: 
        #     PG.plot_cost()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('hashcomp trainer')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/hashcomp_output')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/hashcomp_output')
    parser.add_argument(
        '--episodes',
        type=int,
        default=10000)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01)
    parser.add_argument(
        '--reward-decay',
        type=float,
        default=0.95)
    parser.add_argument(
        '--restore',
        default=False,
        action='store_true')
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=1)
    parser.add_argument(
        '--render',
        default=True,
        action='store_true')
    # parser.add_argument(
    #     '--laziness',
    #     type=float,
    #     default=0.01)
    args = parser.parse_args()

    # save all checkpoints
    args.max_to_keep = args.episodes // args.save_checkpoint_steps

    main(args)