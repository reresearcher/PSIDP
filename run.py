# encoding: utf-8
import os
import argparse
from loguru import logger

import torch
import PSIDP
from model_loader import load_model
from data.data_loader import load_data

label_dim = {
    'flickr25k':24,
}

feature_dim = {
    'vgg16': 4096,
}


def run():
    # Load configuration
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args)

    # Load dataset
    query_dataloader, train_dataloder, retrieval_dataloader = load_data(args.dataset,
                                                                        args.dataset_dir,
                                                                        args.num_query,
                                                                        args.num_train,
                                                                        args.batch_size,
                                                                        args.num_workers,
                                                                        )


    if args.train:
        PSIDP.train(
            args.near_neighbor,
            args.num_train,
            args.batch_size,
            args.dataset,
            train_dataloder,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            feature_dim[args.arch],
            label_dim[args.dataset],
            args.alpha,
            args.beta,
            args.gamma,
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.evaluate_interval,
            args.snapshot_interval,
            args.topk,
        )
    elif args.evaluate:
        model = load_model(args.arch, args.code_length)
        model.load_snapshot(args.checkpoint)
        model.to(args.device)
        model.eval()
        mAP = PSIDP.evaluate(
            model,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            label_dim[args.dataset],
            args.device,
            args.topk,
            )
        logger.info('[Inference map:{:.4f}]'.format(mAP))
    else:
        raise ValueError('Error configuration')


def load_config():

    parser = argparse.ArgumentParser(description='PSIDP')

    parser.add_argument('--device', default=0, type=int,
                        help='GPU node ID')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of loading data threads.')

    parser.add_argument('--arch', default='vgg16', type=str,
                        help='CNN architecture.')
    parser.add_argument('--dataset', default='flickr25k', type=str,
                        help='Dataset name.')
    parser.add_argument('--dataset-dir', default='./Datasets/flickr25k/', type=str,
                        help='Dataset path.')
    parser.add_argument('--code-length', default=16, type=int,
                        help='Binary hash code length.')

    parser.add_argument('--num-query', default=2000, type=int,
                        help='Number of query data points.')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of training data points.')
    parser.add_argument('--topk', default=5000, type=int,
                        help='Calculate map of top k.')

    parser.add_argument('--train', action='store_true',
                        help='Training mode.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate mode.')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path of checkpoint.')

    parser.add_argument('-T', '--max-iter', default=80, type=int,
                        help='Number of iterations.')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size.')

    parser.add_argument('-e', '--evaluate-interval', default=5, type=int,
                        help='Interval of evaluation.')
    parser.add_argument('-s', '--snapshot-interval', default=10, type=int,
                        help='Interval of evaluation.')


    parser.add_argument('--alpha', default=2, type=float,
                        help='Hyper-parameter.')
    parser.add_argument('--beta', default=2, type=float,
                        help='Hyper-parameter.')
    parser.add_argument('--gamma', default=0.0001, type=int,
                        help='Hyper-parameter.')
    parser.add_argument('--near_neighbor', default=3, type=int,
                        help='Hyper-parameter.')
    args = parser.parse_args()
    # GPU
    args.device = torch.device("cuda:%d" % args.device)

    return args


if __name__ == '__main__':
    run()