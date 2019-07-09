import argparse
import importlib
import logging
import os
import shutil

import data


# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("AnomalyDetection")


def run(args):
    print("""
    _____________________
   |                     |
   |                     |
   |     BiGAN + WGAN    |
   |                     /
   |____________________/                            
""")

    has_effect = False

    if args.model and args.dataset and args.split:
        try:
            # train file and its name
            mod_name = "{}.{}_{}".format(args.model, args.split, args.dataset)
            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)
            mod.train(args.nb_epochs, args.w, args.m, args.d, args.dataset, args.rd)

        except Exception as e:
            logger.exception(e)
            logger.error("Ceased with some errors.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <model name> {train, test, run}")


def path(model):
    try:
        assert os.path.isdir(model)
        return model
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(model))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GAN-based Anomaly Detector.')
    parser.add_argument('model', nargs="?", default='bigan', type=path, help='the folder name of the model you want to run e.g gan or bigan')
    parser.add_argument('dataset', nargs="?", default='backblaze', choices=['kdd', 'ali', 'backblaze'], help='the name of the dataset you want to run the experiments on')
    parser.add_argument('split', nargs="?", default='run', choices=['run'], help='train the model or evaluate it')
    parser.add_argument('--nb_epochs', nargs="?", default=500, type=int, help='number of epochs you want to train the dataset on')
    parser.add_argument('--m', nargs="?", default='fm', choices=['cross-e', 'fm'], help='mode/method for discriminator loss')
    parser.add_argument('--w', nargs="?", default=0.1, type=float, help='weight for the sum of the mapping loss function')
    parser.add_argument('--d', nargs="?", default=1, type=int, help='degree for the L norm')
    parser.add_argument('--rd', nargs="?", default=36, type=int, help='random_seed')

    run(parser.parse_args())
