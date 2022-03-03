import argparse

from src.train_handler import TrainHandler
from src.config import config 

t_path = f"{config.base_dir}/data/swda/standard/train.json"
d_path = f"{config.base_dir}/data/swda/standard/dev.json"
l_path = f"{config.base_dir}/data/swda/standard/labels.json"

parser = argparse.ArgumentParser(description='Train Dialogue Act Modelling Model')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--exp_name', type=str,         help='name to save the experiment as')
group.add_argument('--temp', action='store_true', help='to save in temp dir', )
parser.add_argument('--device',   default='cuda',  type=str,  help='device to use (cuda, cpu)')

parser.add_argument('--train_path', default=t_path,  type=str,  help='')
parser.add_argument('--dev_path',   default=d_path,  type=str,  help='')
parser.add_argument('--label_path', default=l_path,  type=str,  help='')
parser.add_argument('--num_labels', default=43,      type=str,  help='')

parser.add_argument('--system',      default='led',     type=str,  help='select system (e.g. bert, roberta etc.)')
parser.add_argument('--system_args', default=None,      type=str,  help='select system arguments')
parser.add_argument('--mode',        default='seq2seq', type=str,  help='select model mode (seq2seq, context)')
parser.add_argument('--mode_args',   default=(3,3),     type=int,  help='select model arguments if using context', nargs=2)

parser.add_argument('--lim',     default=None,  type=int,   help='size of data subset to use (for debugging)')
parser.add_argument('--punct',   action='store_true',       help='whether punctuation should be filtered')
parser.add_argument('--action',  action='store_true',       help='whether actions should be filtered')
parser.add_argument('--hes',     action='store_true',       help='whether hesitations should be filtered')

parser.add_argument('--epochs',  default=100,     type=int,   help='numer of epochs to train')
parser.add_argument('--lr',      default=1e-5,  type=float, help='training learning rate')
parser.add_argument('--bsz',     default=4,     type=int,   help='training batch size')

parser.add_argument('--optim',   default='adam', type=str,   help='which optimizer to use (adam, adamw)')
parser.add_argument('--sched',   default=None,   type=str,   help='which scheduler to use (triangle, exponential, step)')
parser.add_argument('--s_args',  default=None,   type=list,  help='scheduler arguments to use (depends on scheduler)')

parser.add_argument('--print_len', default=100,   type=int,  help='logging training print size')
parser.add_argument('--max_len',   default=None,  type=int,  help='training print size for logging')
parser.add_argument('--class_red', action='store_true',      help='')

if __name__ == '__main__':
    args = parser.parse_args()
    trainer = TrainHandler()
    trainer.train(args)
