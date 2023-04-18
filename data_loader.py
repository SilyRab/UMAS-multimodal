# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/20 13:40
@Auth ： Zhou Ru
"""
import re
from collections import Counter
import numpy as np
from torch.utils.data import TensorDataset
from spacy.lang.en.tag_map import TAG_MAP
import en_core_web_sm
import torch
from gensim.models import word2vec
import random
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# 处理单词，返回单词的合适形式
def preprocess_word(word):
    """
    - Do lowercase
    - Regular expression (number, url, hashtag, user)
        - https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    :param word: str
    :return: word: str
    """
    number_re = r"[-+]?[.\d]*[\d]+[:,.\d]*"
    url_re = r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"
    hashtag_re = r"#\S+"
    user_re = r"@\w+"


    if re.compile(number_re).match(word):
        word = '<NUMBER>'
    elif re.compile(url_re).match(word):
        word = '<URL>'
    elif re.compile(hashtag_re).match(word):
        word = word[1:]  # only erase `#` at the front
    elif re.compile(user_re).match(word):
        word = word[1:]  # only erase `@` at the front
    word = word.lower()
    return word


# 读取所有的句子，返回所有的句子，句子的最大长度、单词的最大长度
def load_sentence(train_path,dev_path,test_path):
    all_sentences = []
    sent_maxlen = 0
    word_maxlen = 0

    for filepath in (train_path,dev_path,test_path):
        #读取句子
        sentences=open(filepath+"/sentence.txt",'r',encoding='utf-8').readlines()
        all_sentences.extend(sentences)

        # 记录最长的句子和单词
        for sentence in sentences:
            sentence = sentence.strip().split()
            sent_maxlen=max(len(sentence),sent_maxlen)
            for word in sentence:
                word=preprocess_word(word)
                word_maxlen=max(len(word),word_maxlen)
    return all_sentences,sent_maxlen,word_maxlen


# 构建词典
def build_vocab(dir,sentences):
    words,chars=[],[]
    for sentence in sentences:
        sentence=sentence.strip().split()
        for word in sentence:
            word=preprocess_word(word)
            words.append(word)
            for char in word:
                chars.append(char)

    word_vocab_path=dir+'/word_vocab'
    char_vocab_path=dir+'/char_vocab'

    word_counts = Counter(words)
    word_vocab_inv=[x[0] for x in word_counts.most_common()]
    word_vocab={}
    word_vocab["[pad]"] = 0
    word_vocab["[unk]"] = 1
    word_vocab["[ttttt]"]=2
    for i,x in enumerate(word_vocab_inv):
        word_vocab[x]=i+3
    id_to_word_vocab={i:x for x,i in word_vocab.items()}

    with open(word_vocab_path, 'w', encoding='utf-8') as f:
        for word in word_vocab:
            f.write(word + "\n")

    char_counts = Counter(chars)
    char_vocab_inv=[x[0] for x in char_counts.most_common()]
    char_vocab={}
    char_vocab["[pad]"] = 0
    char_vocab["[unk]"] = 1
    for i, x in enumerate(char_vocab_inv):
        char_vocab[x]=i+2
    id_to_char_vocab = {i: x for x, i in char_vocab.items()}

    with open(char_vocab_path, 'w', encoding='utf-8') as f:
        for char in char_vocab:
            f.write(char + "\n")
    return word_vocab,char_vocab,id_to_word_vocab,id_to_char_vocab


# 加载词向量矩阵
'''
build=True时重新根据词典生成词向量矩阵
build=false时从磁盘加载词向量矩阵
'''
def load_word_matrix(dir,vocabulary, size=200, build=False, write_word_not_in_model=False):
    word_matrix = np.zeros((len(vocabulary)+1,size), dtype='float32')
    print('word matrix size should be:',len(word_matrix))
    if build:
        model = word2vec.Word2Vec.load('./word2vec/word2vec_200dim.model')
        b = 0
        wordNotinModel = []
        for i, word in vocabulary.items():
            try:
                word_matrix[i]=model[word]
            except KeyError:
                # if a word is not include in the vocabulary, it's word embedding will be set by random.
                word_matrix[i] = np.random.uniform(-0.25,0.25,size)
                b+=1
                wordNotinModel.append(word)
        print('there are %d words not in model'%b)
        if write_word_not_in_model:
            with open(dir+'/word_not_in_model.txt','w',encoding='utf-8') as f:
                for word in wordNotinModel:
                    f.write(word+'\r')
        word_matrix=np.array(word_matrix)
        np.save(dir+'/word_matrix.npy',word_matrix)

    else:
        word_matrix=np.load(dir+'/word_matrix.npy',allow_pickle=True)
        word_matrix = torch.from_numpy(word_matrix)
    print('real word matrix size:',len(word_matrix))
    return word_matrix


def load_matrix_for_res(dir,vocabulary, size=300, build=False, write_word_not_in_model=False):
    word_matrix = np.zeros((len(vocabulary) + 1, size), dtype='float32')
    print('word matrix size should be:', len(word_matrix))
    if build:
        vec_path = '/home/zhouru/zhouru/UMAS-old/data/res14/glove_embedding.npy'
        w2v = np.load(vec_path)
        dict = eval(open('/home/zhouru/zhouru/UMAS-old/data/res14/word2id.txt', 'r').read())
        b = 0
        wordNotinModel = []
        for i, word in vocabulary.items():
            try:
                index=dict[word]
                word_matrix[i] =w2v[index]
            except KeyError:
                # if a word is not include in the vocabulary, it's word embedding will be set by random.
                word_matrix[i] = np.random.uniform(-0.25, 0.25, size)
                b += 1
                wordNotinModel.append(word)
        print('there are %d words not in model' % b)
        if write_word_not_in_model:
            with open(dir + '/word_not_in_model.txt', 'w', encoding='utf-8') as f:
                for word in wordNotinModel:
                    f.write(word + '\r')
        word_matrix = np.array(word_matrix)
        np.save(dir + '/word_matrix.npy', word_matrix)

    else:
        word_matrix = np.load(dir + '/word_matrix.npy', allow_pickle=True)
        word_matrix = torch.from_numpy(word_matrix)
    print('real word matrix size:', len(word_matrix))
    return word_matrix
    print(dict)
    print(len(dict))

'''
将数据转换成向量特征
load_from_disk 是指pos_ids 是否从磁盘加载
'''
def convert_examples_to_features(args,dataset_name,word_vocab,char_vocab,word_maxlen,sent_maxlen,load_from_disk=False):
    dir = './data/'+dataset_name
    print(dir)
    # if args.mode=='ner':
    #     dir = './data/SA/' + dataset_name
    # elif args.mode=='sa':
    #     dir = './data/SA/' + dataset_name


    label = args.NER_label_lst
    features=[]
    if args.no_visual==False:
        img_ids=open(dir+'/img.txt','r').readlines()
        img_features = torch.load('/home/zhouru/zhouru/UMAS/img_vgg_features.pt')
    aspects=open(dir+'/aspect.txt','r').readlines()
    sentences=open(dir+'/sentence.txt','r',encoding='utf-8').readlines()

    if args.mode=='sa' :
        polarities = open(dir + '/polarity.txt', 'r').readlines()

    # 词性标注标签
    pos_map = {tag: i for i, tag in enumerate(TAG_MAP.keys(), 1)}  # Convert to tag -> id
    pos_map["<pad>"] = 0
    pos_pad_idx=0
    nlp = en_core_web_sm.load()

    # 实体标签
    label_vocab = dict()
    for idx, label in enumerate(label):
        label_vocab[label] = idx

    word_pad_idx, char_pad_idx, label_pad_idx = word_vocab['[pad]'], char_vocab['[pad]'], label_vocab['O']
    word_unk_idx, char_unk_idx = word_vocab['[unk]'], char_vocab['[unk]']

    for i in range(len(sentences)):
        assert len(sentences[i].split())==len(aspects[i].split()),"{}-sentence {} and aspects {} have different length.".format(dataset,i,i)
        # print(i)
        # print(polarities[i].strip())

        if args.no_visual == False:
            # 图像特征
            img_ids[i] = int(img_ids[i].strip()[6:])
            img_feature=img_features[img_ids[i]]
        else:
            img_feature=torch.tensor([0,0,0])

        # tokens
        word_ids = []
        char_ids = []
        pos_ids = []
        label_ids = []
        polarity_label_ids=None
        for word in sentences[i].split():
            word = preprocess_word(word)
            word_ids.append(word_vocab.get(word,word_unk_idx))
            if load_from_disk==False:
                pos=nlp(word)[0].tag_
                pos_ids.append(pos_map[pos])

            char_in_word=[]
            for char in word:
                char_in_word.append(char_vocab.get(char,char_unk_idx))
            # 填充字符
            char_padding_len=word_maxlen-len(char_in_word)
            char_in_word+=([char_pad_idx]*char_padding_len)
            char_in_word=char_in_word[:word_maxlen]
            char_ids.append(char_in_word)

        for label in aspects[i].split():
            # if label=='B-OTHER':
            #     label="B-MISC"
            # elif label=="I-OTHER":
            #     label='I-MISC'
            label_ids.append(label_vocab.get(label))


        word_padding_len = sent_maxlen - len(word_ids)
        if args.mode=='sa':
            assert len(sentences[i].split()) == len(polarities[i].split()), "{}-sentence {} and polarities {} have different length.".format(dataset, i, i)
            if polarity_label_ids==None:
                polarity_label_ids=[]
            tmp_polarity = polarities[i].split()
            for polarity_label in tmp_polarity:
                polarity_label_ids.append(int(polarity_label))
            polarity_label_ids += [0] * word_padding_len  # 0为background，没有情感
            polarity_label_ids = polarity_label_ids[:sent_maxlen]
            assert len(polarity_label_ids) == sent_maxlen,\
                    "Error with polarity_label_ids length {} vs {} in sentence {} {}"\
                        .format(len(polarity_label_ids), sent_maxlen, i, sentences[i])

        # 填充单词和标签
        mask = [1] * len(word_ids)
        word_ids+=[word_pad_idx]*word_padding_len
        label_ids+=[label_pad_idx]*word_padding_len
        mask+=[0]*word_padding_len
        for ii in range(word_padding_len):
            char_ids.append([char_pad_idx]*word_maxlen)

        word_ids=word_ids[:sent_maxlen]
        label_ids=label_ids[:sent_maxlen]
        char_ids=char_ids[:sent_maxlen]
        mask=mask[:sent_maxlen]

        if load_from_disk == False:
            pos_ids += [pos_pad_idx] * word_padding_len
            pos_ids=pos_ids[:sent_maxlen]
            assert len(pos_ids) == sent_maxlen, "Error with pos_ids length {} vs {}".format(len(pos_ids), sent_maxlen)

        assert len(word_ids) == sent_maxlen, "Error with word_ids length {} vs {}".format(len(word_ids), sent_maxlen)
        assert len(char_ids) == sent_maxlen, "Error with char_ids length {} vs {}".format(len(char_ids), sent_maxlen)
        assert len(label_ids) == sent_maxlen, "Error with label_ids length {} vs {}".format(len(label_ids), sent_maxlen)
        assert len(mask) == sent_maxlen, "Error with mask length {} vs {}".format(len(mask), sent_maxlen)

        if load_from_disk==False:
            features.append([word_ids, char_ids, img_feature, mask, label_ids, polarity_label_ids, pos_ids])
        else:
            features.append([word_ids, char_ids, img_feature, mask, label_ids, polarity_label_ids])

    # Convert to Tensors and build dataset
    all_word_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_char_ids = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_img_feature = torch.stack([f[2] for f in features])
    all_mask = torch.tensor([f[3] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[4] for f in features], dtype=torch.long)

    if load_from_disk == False:
        all_pos_ids = torch.tensor([f[6] for f in features], dtype=torch.long)
        np.savez(dir+'/' + 'pos_ids.npz', pos_ids=all_pos_ids)
    else:
        all_pos_ids = np.load(dir+'/' + "pos_ids.npz", allow_pickle=True)['pos_ids'][()]
        all_pos_ids=torch.tensor(all_pos_ids,dtype=torch.long)

    if args.mode == 'sa':
        all_polarity_label_ids = torch.tensor([f[5] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_word_ids, all_pos_ids, all_char_ids, all_img_feature, all_mask, all_label_ids,
                                all_polarity_label_ids)
        return dataset
    else:
        dataset = TensorDataset(all_word_ids,all_pos_ids, all_char_ids, all_img_feature, all_mask, all_label_ids)
        return dataset


if __name__=='__main__':
    train_path = './data/lap14/train'
    dev_path = './data/lap14/dev'
    test_path = './data/lap14/test'
    sentences, sent_maxlen, word_maxlen = load_sentence(train_path, dev_path, test_path)
    word_vocab, char_vocab, id_to_word_vocab, id_to_char_vocab = build_vocab('./data/lap14',sentences)
    matrix=load_word_matrix('./data/lap14',id_to_word_vocab, build=True, write_word_not_in_model=True)

    features = convert_examples_to_features('sa','test', word_vocab, char_vocab, word_maxlen, sent_maxlen)
    print('load test data done.')
    features = convert_examples_to_features('sa','dev', word_vocab, char_vocab, word_maxlen, sent_maxlen)
    print('load dev data done.')
    features = convert_examples_to_features('sa','train-over', word_vocab, char_vocab, word_maxlen, sent_maxlen)
    print('load train data done.')