# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/20 13:42
@Auth ： Zhou Ru
"""
import numpy
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam, RMSprop
from data_loader import *
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from torchcrf import CRF
from AE_and_SA import UMAS
from utils import init_logger,random_weight
import time
import os
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pdb

OPTIMIZER_LIST = {
    "adam": Adam,
    "rmsprop": RMSprop
}

def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

def compute_metrics(labels, preds):
    assert len(labels) == len(preds)
    return f1_pre_rec(labels, preds)

def report(labels, preds):
    return classification_report(labels, preds)


class Trainer(object):
    def __init__(self, args,word_vocab, char_vocab, id_to_word_vocab, id_to_char_vocab,train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.NER_label_lst=self.args.NER_label_lst
        self.NER_begin=self.NER_label_lst.index('B')
        self.pad_token_label_id = 0
        self.word_vocab, self.char_vocab, self.word_ids_to_tokens, self.char_ids_to_tokens = word_vocab, char_vocab, id_to_word_vocab, id_to_char_vocab
        # if self.args.dataset=='SA':
        self.pretrained_word_matrix = load_word_matrix(args.data_dir,self.word_ids_to_tokens,build=False,write_word_not_in_model=False)
        # else:
        #     self.pretrained_word_matrix = load_matrix_for_res(args.data_dir,self.word_ids_to_tokens,build=False,write_word_not_in_model=False)


        self.model = UMAS(args, self.pretrained_word_matrix)
        self.crf1 = CRF(num_tags=self.args.entity_num, batch_first=True)
        self.crf2 = CRF(num_tags=self.args.polarity_num, batch_first=True)

        # GPU or CPU
        # self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.device="cpu"
        self.args.device=self.device
        self.model.to(self.device)

        self.opinion_list =[19, 20, 21, 33, 34, 35, 50, 41, 42, 43, 44, 45, 46, 55, 56, 49, 11, 18, 23]
        self.logger,self.log_num=init_logger()
        # log文件写入参数
        options=vars(args)
        for k,v in options.items():
            self.logger.info('{},{}'.format(k,v))
        self.logger.info('ner_labels: {}'.format(self.NER_label_lst))
        self.logger.info('ner_labels_num: {}'.format(self.args.entity_num))
        self.args.logger=self.logger
        # 0情感的随机分配
        self.weight_data = {1: 113, 2: 607 + 100, 3: 317}


    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        self.logger.info("***** Running evaluation on %s dataset *****", mode)
        self.logger.info("  Num examples = %d", len(dataset))
        self.logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        entity_preds = None
        out_label_ids = None
        polarity_preds = None
        out_polarity_ids = None
        criterion=torch.nn.CrossEntropyLoss(reduction='mean')

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'pos_ids': batch[1],
                          'char_ids': batch[2],
                          'img_feature': batch[3],
                          }
                mask = batch[4]
                label_ids = batch[5]
                polarity_ids=None
                if self.args.sa==True:
                    polarity_ids=batch[6]
                logits = self.model(**inputs)
                tmp_eval_loss1,tmp_eval_loss2=0,0
                if self.args.ner:
                    tmp_eval_loss1= self.crf1(logits[0], label_ids, mask.byte(), reduction='mean')
                    tmp_eval_loss1 = tmp_eval_loss1 * -1  # negative log likelihood
                if self.args.sa:
                    tmp_eval_loss2 = criterion(logits[1].view(-1,self.args.polarity_num), polarity_ids.view(-1))
                    # tmp_eval_loss2 = tmp_eval_loss2 * -1  # negative log likelihood
                tmp_eval_loss=tmp_eval_loss1+tmp_eval_loss2
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if out_label_ids is None:
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

            if self.args.ner:
                if entity_preds is None:
                    entity_preds = np.array(self.crf1.decode(logits[0], mask=mask.byte()))
                else:
                    entity_preds = np.append(entity_preds, np.array(self.crf1.decode(logits[0], mask=mask.byte())), axis=0)

            if self.args.sa:
                # 修正预测将实体预测为0的
                tmp = torch.ones(self.args.polarity_num)
                tmp[0] = 0
                polarity_mask = tmp.repeat(self.args.max_seq_len, 1)
                polarity_mask = polarity_mask.repeat(len(mask), 1, 1)
                tmp = torch.mul(logits[1].detach().cpu(), polarity_mask)
                # print(tmp)
                if polarity_preds is None:
                    polarity_preds = np.array(torch.argmax(tmp,dim=-1))
                    out_polarity_ids = polarity_ids.detach().cpu().numpy()
                else:
                    polarity_preds = np.append(polarity_preds, np.array(torch.argmax(tmp,dim=-1)), axis=0)
                    out_polarity_ids = np.append(out_polarity_ids, polarity_ids.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        if self.args.ner:
            # Slot result
            slot_label_map = {i: label for i, label in enumerate(self.NER_label_lst)}
            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            entity_preds_list = [[] for _ in range(out_label_ids.shape[0])]

            for i in range(len(entity_preds)):
                for j in range(len(entity_preds[i])):
                    # if out_label_ids[i, j] != self.pad_token_label_id:
                    if self.args.ner:
                        out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                        entity_preds_list[i].append(slot_label_map[entity_preds[i][j]])


        if self.args.sa:
            out_polarity_list = [[] for _ in range(out_polarity_ids.shape[0])]
            polarity_preds_list = [[] for _ in range(out_polarity_ids.shape[0])]


            for i in range(len(out_label_ids)):
                for j in range(len(out_label_ids[i])):
                    # 将实体首个单词的情感预测为整个实体的情感
                    if out_label_ids[i, j]==self.NER_begin:
                        out_polarity_list[i].append(out_polarity_ids[i][j])
                        p_tmp=polarity_preds[i][j]
                        # p_tmp=p_tmp if p_tmp!=0 else random_weight(self.weight_data)
                        p_tmp=p_tmp if p_tmp!=0 else 2
                        polarity_preds_list[i].append(p_tmp)

        ner_f1,sa_f1,aesa_f1=0,0,0
        ner_p,sa_acc=0,0
        if self.args.aesa and self.args.aesa_eval:
            tp = 0  # 实际预测正确的量，aspect和sentiment 同时正确
            pred_total = 0  # precision 的分母  预测为实体的
            num_total = 0  # recall的分母  实际为实体

            for i in range(len(entity_preds_list)):
                sent_aspect_num = 0
                for j in range(len(entity_preds_list[i])):
                    if out_label_list[i][j] == 'B' and entity_preds_list[i][j]=='B':
                        num_total += 1
                        pred_total += 1
                        sent_aspect_num += 1
                        matched = True
                        for k in range(j+1, len(entity_preds_list[i])):
                            if out_label_list[i][k] != 'I':
                                break
                            elif out_label_list[i][k] != entity_preds_list[i][k]:
                                matched = False
                        # 实体预测正确  情感正确
                        if matched and out_polarity_list[i][sent_aspect_num - 1] == polarity_preds_list[i][sent_aspect_num - 1]:
                            tp += 1
                        #else:# 实体预测错误 或 情感预测错误
                    elif out_label_list[i][j]=='B' and entity_preds_list[i][j]!='B':
                        sent_aspect_num+=1
                        num_total+=1
                    # 将不是实体的预测成实体
                    elif out_label_list[i][j] != 'B' and entity_preds_list[i][j] == 'B':
                        pred_total += 1
                assert sent_aspect_num == len(out_polarity_list[i]),self.logger.info('sent_aspect_num:{},true_num:{}'.format(sent_aspect_num,len(out_polarity_list[i])))
            self.logger.info("tp:{}".format(tp))
            self.logger.info("total_num:{}".format(num_total))
            self.logger.info("pred_num:{}".format(pred_total))

            aeas_p = tp / pred_total * 100 if pred_total>0 else 0
            aeas_r = tp / num_total * 100
            aeas_f1 = 2 * aeas_p * aeas_r / (aeas_p + aeas_r) if (aeas_p + aeas_r)>0 else 0
            aesa_f1=aeas_f1
            self.logger.info("aeas.precision:{}".format(aeas_p))
            self.logger.info("aeas.recall:{}".format(aeas_r))
            self.logger.info("aeas.f1:{}".format(aeas_f1))
        return ner_f1,sa_f1,ner_p,sa_acc,aesa_f1


    def save_model(self,suffix):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        # Save argument
        torch.save(self.args, os.path.join(self.args.model_dir, 'args_'+suffix+'.pt'))
        # Save model for inference
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, 'model_'+suffix+'.pt'))
        self.logger.info("Saving model checkpoint to {}".format(os.path.join(self.args.model_dir, 'model_'+suffix+'.pt')))


    def load_model(self,model_file,args_file=None):
        # Check whether model exists
        if args_file !=None:
            self.args = torch.load(args_file)
            self.logger.info("***** Args loaded *****")
        self.model.load_state_dict(torch.load(model_file))
        self.model.to(self.device)
        self.logger.info("***** model {} Loaded *****".format(model_file))