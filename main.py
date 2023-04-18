# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/20 13:43
@Auth ： Zhou Ru
"""

import argparse
from trainer import Trainer, OPTIMIZER_LIST
from data_loader import *
from utils import *


def main(args):
    set_seed(args)
    train_path = args.data_dir + '/train'
    dev_path = args.data_dir + '/dev'
    test_path = args.data_dir + '/test'
    sentences, sent_maxlen, word_maxlen = load_sentence(train_path, dev_path, test_path)
    args.max_seq_len = sent_maxlen
    args.max_word_len = word_maxlen
    word_vocab, char_vocab, id_to_word_vocab, id_to_char_vocab = build_vocab(args.data_dir, sentences)
    args.word_vocab_size = len(word_vocab)
    args.char_vocab_size = len(char_vocab)
    # train_dataset = convert_examples_to_features(args, 'train', word_vocab, char_vocab, word_maxlen,sent_maxlen, load_from_disk=True)
    # print('load train data done.')
    # dev_dataset = convert_examples_to_features(args, 'dev', word_vocab, char_vocab, word_maxlen, sent_maxlen,load_from_disk=True)
    # print('load dev data done.')
    test_dataset = convert_examples_to_features(args, 'test', word_vocab, char_vocab, word_maxlen, sent_maxlen,load_from_disk=True)
    print('load test data done.')
    trainer = Trainer(args, word_vocab, char_vocab, id_to_word_vocab, id_to_char_vocab, test_dataset=test_dataset)

    mf='./model/model.pt'
    af='./model/args.pt'

    trainer.load_model(mf, af)
    trainer.evaluate('test')


import datetime
if __name__ == '__main__':
    start=datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")

    parser.add_argument("--train_file", default="train", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test", type=str, help="Test file")

    parser.add_argument("--max_seq_len", default=40, type=int, help="Max sentence length")
    parser.add_argument("--max_word_len", default=29, type=int, help="Max word length")

    parser.add_argument("--word_vocab_size", default=12486, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=52, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=200, type=int, help="Word embedding size")
    parser.add_argument("--char_emb_dim", default=30, type=int, help="Character embedding size")
    parser.add_argument("--final_char_dim", default=50, type=int, help="Dimension of character cnn output")
    parser.add_argument("--ner_hidden_dim", default=200, type=int, help="Dimension of BiLSTM output, att layer (denoted as k) etc.")
    parser.add_argument("--sa_hidden_dim", default=100, type=int, help="Dimension of BiLSTM output, att layer (denoted as k) etc.")
    parser.add_argument("--initial_visual_dim", default=512, type=int, help="initial dimension of pictures")
    parser.add_argument("--pos_emb_dim", default=16, type=int, help="pos embedding size")
    parser.add_argument("--pos_att_head", default=4, type=int, help="pos embedding size")


    parser.add_argument("--kernel_lst", default="2,3,4", type=str, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=7, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=15, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation")
    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate")
    parser.add_argument("--ner_learning_rate", default=0.001, type=float, help="The initial learning rate")
    parser.add_argument("--sa_learning_rate", default=0.001, type=float, help="The initial learning rate")
    parser.add_argument("--different_learning_rate", action="store_true",default=False, help="Whether to use different learning rate for sa and ner.")


    parser.add_argument("--dropout_rate", default=0.25, type=float, help="The initial dropout rate")

    parser.add_argument("--num_train_epochs", default=50, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--logging_steps', type=int, default=30, help="Log every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--sa_layer', type=int, default=1, help="num of sa_stack layer")

    parser.add_argument("--weight1", default=[0,0.85,0.2,0.3], type=list, help="loss weight for sa")
    parser.add_argument("--weight2", default=[0,1,0.1,0.45], type=list, help="loss weight for opinion")
    parser.add_argument("--private_conv", action="store_true",default=False, help="Whether to use CNN for private features.")
    parser.add_argument("--private_conv_kernel_size", type=int, default=3,  help="Whether to use CNN for private features.")


    parser.add_argument("--add_pos", action="store_true",default=True, help="Whether to use pos features.")
    parser.add_argument("--ner_no_grad", action="store_true",default=False, help="Whether to use different learning rate for sa and ner.")
    parser.add_argument("--ner", action="store_true", default=True, help="Whether to run NER training and evaluate.")
    parser.add_argument("--sa", action="store_true", default=True, help="Whether to run SA training and evaluate.")
    parser.add_argument("--aesa", action="store_true", default=True, help="Whether to run SA training and evaluate.")
    parser.add_argument("--ner_eval", action="store_true", default=True, help="Whether to run NER training and evaluate.")
    parser.add_argument("--sa_eval", action="store_true", default=True,help="Whether to run SA training and evaluate.")
    parser.add_argument("--aesa_eval", action="store_true", default=True,help="Whether to run SA training and evaluate.")
    parser.add_argument("--select", default='aesa',type=str,help="--")
    parser.add_argument("--loss_weight", default=[1,1], type=list, help="loss weight for ae,sa,os")
    parser.add_argument("--load_trained_model", action="store_true", default=False,help="Whether to load trained model to continue training.")
    parser.add_argument("--trained_model_suffix", default="aesa3", type=str, help="--")

    parser.add_argument("--no_visual", action="store_true", default=False, help="--")
    parser.add_argument("--no_opinion_att", action="store_true", default=False, help="--")
    parser.add_argument("--no_sa_selfatt", action="store_true", default=False, help="--")
    parser.add_argument("--dataset", default="SA", type=str, help="--")
    parser.add_argument("--ner_grad", action="store_true", default=True, help="--")
    parser.add_argument("--sa_grad", action="store_true", default=False, help="--")

    args = parser.parse_args()
    args.polarities = [0, 1, 2, 3]
    args.sa_weight = {'SA':[0,0.85,0.2,0.3]}

    if args.sa==True:
        args.mode='sa'
    elif args.ner==True:
        args.mode='ner'
    args.NER_label_lst = ["O", "B", "I"]

    args.entity_num=len(args.NER_label_lst)
    args.polarity_num=len(args.polarities)

    # For 16VGG img features (DO NOT change this part)
    args.num_img_region = 49
    args.img_feat_dim = 512
    main(args)
    end=datetime.datetime.now()
    r=end-start
    print(r.seconds)