import argparse
from ccsdg.utils.file_utils import gen_random_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="unet", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--log_folder', required=True,
                        help='Log folder.')
    parser.add_argument('--tag', default="{}".format(gen_random_str()), required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--save_interval', type=int, default=25, required=False,
                        help='save_interval.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--no_shuffle', default=False, required=False, action='store_true',
                        help="No shuffle training set.")
    parser.add_argument('--num_threads', type=int, default=4, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', required=True,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        required=True, help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=True, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')

    args = parser.parse_args()
    
  
    model_name = args.model



    if model_name == 'unet':
        from ccsdg.training.train_nets.train_unet import train
    elif model_name == 'unet_ccsdg':
        from ccsdg.training.train_nets.train_unet_ccsdg import train
    else:
        print('No model named "{}"!'.format(model_name))
        return
    train(args)


if __name__ == '__main__':
    main()
