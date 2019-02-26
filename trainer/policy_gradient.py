import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

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

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate,
        reward_decay,
        output_dir,
        max_to_keep,
        restore,
        save_checkpoint_steps
    ):
        self.sess = tf.Session()
        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay
        self.output_dir = output_dir
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        self.build_network()
        self.cost_history = []

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        summary_path = os.path.join(self.output_dir, 'summary')
        self.writer = tf.summary.FileWriter(summary_path, self.sess.graph)
        self.save_checkpoint_steps = save_checkpoint_steps

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()
        # self.saver = tf.train.Saver(max_to_keep=max_to_keep)

        self.sess.run(tf.global_variables_initializer())

        # Restore model
        if restore:
            # self.output_dir = output_dir
            # self.restore_path = tf.train.latest_checkpoint(self.output_dir)
            # self.output_dir = output_dir
            # self.saver.restore(self.sess, self.output_dir)
            restore_path = tf.train.latest_checkpoint(self.output_dir)
            print('Restoring from {}'.format(restore_path))
            self.saver.restore(self.sess, restore_path)

    def store_transition(self, s, a, r):
        """
            Store play memory for training
            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)


    def choose_action(self, observation):
        """
            Choose action based on observation
            Arguments:
                observation: array of state, has shape (num_features)
            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        # print(observation)
        observation = preprocess(observation)
        observation = observation[:, np.newaxis]
        
        #observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        _, summaries, global_step = self.sess.run([self.train_op, self.summaries, self.global_step], feed_dict={
             self.X: np.vstack(self.episode_observations).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })
        self.writer.add_summary(summaries, global_step)
        self.writer.flush()

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        # print(self.global_step, self.save_checkpoint_steps)
        # if self.global_step % self.save_checkpoint_steps == 0:
        #     print('hit\n\n\n')
        output_dir = os.path.join(self.output_dir, 'model.ckpt')
        output_dir = self.saver.save(self.sess, output_dir, global_step=self.global_step)
        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards, dtype=float)
        #print("DISCOUNTED\n", discounted_episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
         #   print("CUMULATIVE :\n", cumulative)
            discounted_episode_rewards[t] = cumulative
          #  print("Ri :\n", discounted_episode_rewards[t])

        #print("DISCOUNTED\n", discounted_episode_rewards)
        #print("MEAN\n", np.mean(discounted_episode_rewards))
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 10
        units_layer_2 = 10
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss
            tf.summary.scalar('loss', loss)
        self.summaries = tf.summary.merge_all()
        with tf.name_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, global_step=self.global_step)

    def plot_cost(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
