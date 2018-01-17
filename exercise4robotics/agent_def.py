""" Create a agent """
from utils     import Options
from model import DQLAgent, cnn_factory


opt = Options()
state_with_history_dim = opt.hist_len * opt.state_siz

q_fn = cnn_factory(state_with_history_dim, filters=[8, 16, 16, 32],
                   kernels_size=[64, 32, 16, 4], hidden_units=[],
		   output_units=opt.act_num)
agent = DQLAgent(q_fn, opt.act_num, model_dir=opt.checkpoint_dir,
                 learning_rate=opt.learning_rat,
                 discount=opt.q_loss_discount,
                 epsilon=opt.policy_eps, epsilon_min=opt.policy_eps_min,
                 epsilon_decay_interval=opt.steps//opt.train_interval//2)
