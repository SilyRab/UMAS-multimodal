# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertSelfAttention,BertConfig
from spacy.lang.en.tag_map import TAG_MAP
from data_loader import *
from utils import init_logger



'''
字符级特征表示
'''
class CharCNN(nn.Module):
    def __init__(self,
                 max_word_len=30,
                 kernel_lst="2,3,4",
                 num_filters=32,
                 char_vocab_size=1000,
                 char_emb_dim=30,
                 final_char_dim=50):
        super(CharCNN, self).__init__()

        # 初始化字符嵌入式表示
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.25, 0.25)

       # 卷积操作
        kernel_lst = list(map(int, kernel_lst.split(",")))
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size // 2),
                nn.Tanh(),
                nn.MaxPool1d(max_word_len - kernel_size + 1),
                nn.Dropout(0.25)
            ) for kernel_size in kernel_lst
        ])

        # 卷积结果拼接
        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(kernel_lst), 100),
            nn.ReLU(),  # As same as the original code implementation
            nn.Dropout(0.25),
            nn.Linear(100, final_char_dim)
        )

    def forward(self, x):
        """
        :param x: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, final_char_dim)
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_emb(x)  # (b, s, w, d)
        x = x.view(batch_size * max_seq_len, max_word_len, -1)  # (b*s, w, d)
        x = x.transpose(2, 1)  # (b*s, d, w): Conv1d takes in (batch, dim, seq_len), but raw embedded is (batch, seq_len, dim)

        conv_lst = [conv(x) for conv in self.convs]
        conv_concat = torch.cat(conv_lst, dim=-1)  # (b*s, num_filters, len(kernel_lst))
        conv_concat = conv_concat.view(conv_concat.size(0), -1)  # (b*s, num_filters * len(kernel_lst))

        output = self.linear(conv_concat)  # (b*s, final_char_dim)
        output = output.view(batch_size, max_seq_len, -1)  # (b, s, final_char_dim)
        return output


'''
词性特征表示
'''
class POS_embedding(nn.Module):
    def __init__(self,args,pos_embedding_size=32):
        super(POS_embedding, self).__init__()
        self.args=args
        # define POS tagging
        self.tag_map = {tag: i for i, tag in enumerate(TAG_MAP.keys(), 1)}  # Convert to tag -> id
        self.tag_map["<pad>"] = 0
        self.embeddings = nn.Embedding(len(self.tag_map), pos_embedding_size)
        pos_attention_config = BertConfig(hidden_size=pos_embedding_size, num_attention_heads=args.pos_att_head)
        self.pos_attention = BertSelfAttention(pos_attention_config)


    def forward(self,pos_ids):
        attention_mask = torch.ones_like(pos_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        pos_output = self.embeddings(pos_ids)
        pos_output = self.pos_attention(pos_output, extended_attention_mask)[0]
        return pos_output

'''
拼接字符特征表示和词向量特征表示
使用BiLSTM获取隐藏状态
'''
class BiLSTM(nn.Module):
    def __init__(self, args, pretrained_word_matrix):
        super(BiLSTM, self).__init__()
        self.args = args
        self.char_cnn = CharCNN(max_word_len=args.max_word_len,
                                kernel_lst=args.kernel_lst,
                                num_filters=args.num_filters,
                                char_vocab_size=args.char_vocab_size,
                                char_emb_dim=args.char_emb_dim,
                                final_char_dim=args.final_char_dim)
        if pretrained_word_matrix is not None:
            self.word_emb = nn.Embedding.from_pretrained(pretrained_word_matrix)
        else:
            self.word_emb = nn.Embedding(args.word_vocab_size, args.word_emb_dim, padding_idx=0)
            nn.init.uniform_(self.word_emb.weight, -0.25, 0.25)

        self.bi_lstm = nn.LSTM(input_size=args.word_emb_dim+args.final_char_dim,
                               hidden_size=args.ner_hidden_dim//2,  # Bidirectional will double the hidden_size
                               bidirectional=True,
                               batch_first=True)

    def forward(self, word_ids, char_ids):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, dim)
        """
        w_emb = self.word_emb(word_ids)
        c_emb = self.char_cnn(char_ids)
        w_c_emb = torch.cat([w_emb, c_emb], dim=-1)
        lstm_output, _ = self.bi_lstm(w_c_emb, None)
        return lstm_output

'''
获取文本引导的视觉表示
获取视觉引导的文本表示
'''
class CoAttention(nn.Module):
    def __init__(self, args):
        super(CoAttention, self).__init__()
        self.args = args

        # 文本引导的视觉表示——线性层
        self.text_linear_1 = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim, bias=True)
        self.img_linear_1 = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(args.ner_hidden_dim * 2, 1)

        # 视觉引导的文本表示——线性层
        self.text_linear_2 = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim, bias=False)
        self.img_linear_2 = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.ner_hidden_dim * 2, 1)


    def forward(self, text_features, img_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. 文本引导注意的视觉表示 ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = text_features.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)
        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = self.text_linear_1(text_features_rep)
        img_features_rep = self.img_linear_1(img_features_rep)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([text_features_rep, img_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        visual_att = self.att_linear_1(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_img_features = torch.matmul(visual_att, img_features)  # Vt_hat

        ############### 2. 视觉引导注意的文本表示 ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = att_img_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len, 1)
        text_features_rep = text_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = self.img_linear_2(img_features_rep)
        text_features_rep = self.text_linear_2(text_features_rep)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([img_features_rep, text_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        textual_att = self.att_linear_2(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_text_features = torch.matmul(textual_att, text_features)  # Ht_hat

        return att_text_features, att_img_features


'''
多模态融合
'''
class GMF(nn.Module):
    def __init__(self,hidden_dim):
        super(GMF, self).__init__()
        self.hidden_dim=hidden_dim
        self.text_linear = nn.Linear(hidden_dim, hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(hidden_dim, hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, att_text_features, att_img_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

        gate_img = self.gate_linear(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img = torch.sigmoid(gate_img)  # [batch_size, max_seq_len, 1]
        gate_img = gate_img.repeat(1, 1, self.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img, new_img_feat) + torch.mul(1 - gate_img, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features


class Ner_Fusion_Feature(nn.Module):
    def __init__(self,args):
        super(Ner_Fusion_Feature, self).__init__()
        self.args=args
        self.co_attention = CoAttention(args)
        self.gmf = GMF(args.ner_hidden_dim)

        self.text_linear = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim, bias=False)
        self.multimodal_linear = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim, bias=True)
        self.gate_linear = nn.Linear(args.ner_hidden_dim * 2, 1)
        self.resv_linear = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim)


    def forward(self,text_features, img_features):
        # 加入注意力机制的文本和图像表示
        att_text_features, att_img_features = self.co_attention(text_features, img_features)
        # 多模态特征融合
        multimodal_features = self.gmf(att_text_features, att_img_features)
        # 多模态特征与文本特征融合
        concat_feat = torch.cat([self.text_linear(text_features), self.multimodal_linear(multimodal_features)], dim=-1)
        # This part is not written on equation, but if is needed
        filtration_gate = torch.sigmoid(self.gate_linear(concat_feat))  # [batch_size, max_seq_len, 1]
        filtration_gate = filtration_gate.repeat(1, 1, self.args.ner_hidden_dim)  # [batch_size, max_seq_len, hidden_dim]

        reserved_multimodal_feat = torch.mul(filtration_gate,
                                             torch.tanh(self.resv_linear(
                                                 multimodal_features)))  # [batch_size, max_seq_len, hidden_dim]
        return reserved_multimodal_feat



class sentiment_visual(nn.Module):
    def __init__(self,args):
        super(sentiment_visual, self).__init__()
        # 文本引导的视觉表示——线性层
        self.args=args
        self.text_linear = nn.Linear(args.sa_hidden_dim, args.sa_hidden_dim, bias=True)
        self.img_linear = nn.Linear(args.sa_hidden_dim, args.sa_hidden_dim, bias=False)
        self.att_linear = nn.Linear(args.sa_hidden_dim * 2, 1)

    def forward(self,text_features,img_features):
        # 文本引导注意的视觉表示
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = text_features.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)
        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = self.text_linear(text_features_rep)
        img_features_rep = self.img_linear(img_features_rep)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([text_features_rep, img_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        visual_att = self.att_linear(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_img_features = torch.matmul(visual_att, img_features)  # Vt_hat

        return att_img_features



class opinion_attention(nn.Module):
    def __init__(self,args):
        super(opinion_attention, self).__init__()
        self.opinion_linear1 = nn.Linear(args.sa_hidden_dim, args.sa_hidden_dim)
        self.text_linear = nn.Linear(args.sa_hidden_dim, args.sa_hidden_dim, bias=True)
        self.opinion_linear = nn.Linear(args.sa_hidden_dim, args.sa_hidden_dim, bias=False)
        self.att_linear = nn.Linear(args.sa_hidden_dim * 2, 1)
        self.args=args
        self.noun_list = [25, 26, 27, 28]
        self.opinion_list = [19, 20, 21, 33, 34, 35, 41, 42, 43, 44, 45, 46]

    def position_matrix(self,max_len,pos_list):
        a = np.zeros([max_len, max_len], dtype=np.float32)

        for i in range(max_len):
            for j in range(max_len):
                if i == j:
                    a[i][j] = 0.5
                else:
                    a[i][j] = 1 / (np.log2(2 + abs(i - j)))
                    # a[i][j] = 1/(abs(i - j))

        a=torch.from_numpy(np.array(a)).to(self.args.device)
        # 为观点词分配更大的权重
        opinion_att=torch.ones(max_len)
        for i in range(len(pos_list)):
            if pos_list[i] in self.opinion_list:
                opinion_att[i]=8
        opinion_att=opinion_att.unsqueeze(0).repeat(max_len,1)
        # self.args.logger.info('size:{} {}'.format(a.size(),opinion_att.size()))
        a=a*opinion_att
        a=a.numpy()
        return a


    def forward(self,opinion_features,text_features,pos_ids):
        # opinion_features=self.opinion_linear1(opinion_features)
        # 位置信息
        position_m = list()
        for i in range(len(pos_ids)):
            position_m.append(self.position_matrix(self.args.max_seq_len,pos_ids[i]))
        position_m=torch.from_numpy(np.array(position_m)).to(self.args.device)
        # self.args.logger.info('position:{} {}'.format(position_m.size(),position_m))

        # 文本引导的观点词注意力，获得某个单词对应的观点特征
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        text_features_rep = text_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len, 1)
        opinion_features_rep = opinion_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        text_features_rep = self.text_linear(text_features_rep)
        opinion_features_rep = self.opinion_linear(opinion_features_rep)
        concat_features = torch.cat([text_features_rep, opinion_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        opinion_att = self.att_linear(concat_features).squeeze(-1)
        # self.args.logger.info('opinion_att:{} {}'.format(opinion_att.size(),opinion_att))
        opinion_att*=position_m
        opinion_att = torch.softmax(opinion_att, dim=-1)
        # self.args.logger.info('opinion_features:{} {}'.format(opinion_features.size(),opinion_features))

        att_opinion_features = torch.matmul(opinion_att, opinion_features)
        return att_opinion_features



class Sa_Fusion_Feature(nn.Module):
    def __init__(self,args):
        super(Sa_Fusion_Feature, self).__init__()
        self.args=args
        self.sentiment_visual=sentiment_visual(args)
        # 情感模块的自注意表示层
        self.text_attention1 = BertSelfAttention(BertConfig(hidden_size=args.sa_hidden_dim, num_attention_heads=4))
        self.sa_text_linear = nn.Linear(args.sa_hidden_dim*2, args.sa_hidden_dim)
        self.tanh = nn.Tanh()
        self.gmf=GMF(args.sa_hidden_dim)


    def forward(self,sa_text_features,img_features,attention_mask):
        if self.args.no_visual:
            if self.args.no_sa_selfatt==False:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                sa_text_features = self.text_attention1(sa_text_features, extended_attention_mask)[0]
                sa_text_features = self.tanh(sa_text_features)
            return sa_text_features
        else:
            sa_visual=self.sentiment_visual(sa_text_features,img_features)
            if self.args.no_sa_selfatt == False:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                sa_text_features = self.text_attention1(sa_text_features,extended_attention_mask)[0]
                sa_text_features=self.tanh(sa_text_features)
            fusion_feature=self.gmf(sa_text_features,sa_visual)

        # 视觉过滤
        return fusion_feature


class Sa_Stack(nn.Module):
    def __init__(self,args,layer_num=1):
        super(Sa_Stack, self).__init__()
        self.args=args
        self.layer_num=layer_num
        self.sa_fusion_feature = Sa_Fusion_Feature(args)
        self.opinion_attention = opinion_attention(args)
        if self.args.add_pos:
            self.sa_linear1 = nn.Linear(args.sa_hidden_dim * 3+self.args.pos_emb_dim, args.sa_hidden_dim)
            if self.args.no_opinion_att:
                self.sa_linear1 = nn.Linear(args.sa_hidden_dim * 2 + self.args.pos_emb_dim, args.sa_hidden_dim)
        else:
            self.sa_linear1 = nn.Linear(args.sa_hidden_dim * 3, args.sa_hidden_dim)
            if self.args.no_opinion_att:
                self.sa_linear1 = nn.Linear(args.sa_hidden_dim * 2, args.sa_hidden_dim)

        self.sa_linear2 = nn.Linear(args.sa_hidden_dim, args.polarity_num)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.25)

        if self.layer_num>1:
            self.opinion_attention_stack=[opinion_attention(args) for i in range(self.layer_num)]
            self.sa_fusion_feature_stack=[Sa_Fusion_Feature(args) for i in range(self.layer_num)]
            self.sa_linear1_stack=[nn.Linear(args.sa_hidden_dim * 3, args.sa_hidden_dim) for i in range(self.layer_num)]
            self.sa_linear2_stack=[nn.Linear(args.sa_hidden_dim, args.polarity_num) for i in range(self.layer_num)]
            self.sa_linear3_stack=[nn.Linear(args.sa_hidden_dim,self.args.sa_hidden_dim-self.args.polarity_num) for i in range(self.layer_num)]


    def forward(self,sa_text_features,opinion_features,pos_ids,img_features, pos_features):
        assert self.layer_num>=1
        attention_mask = torch.ones_like(pos_ids)
        if self.layer_num==1:
            if self.args.no_opinion_att:
                if self.args.add_pos:
                    sa_fusion_features = self.sa_fusion_feature(sa_text_features, img_features, attention_mask)
                    sa_output = self.sa_linear1(self.dropout(torch.cat([sa_text_features, sa_fusion_features,pos_features], dim=-1)))
                    sa_output = self.sa_linear2(sa_output)
                    sa_output = self.softmax(sa_output)
                else:
                    sa_fusion_features = self.sa_fusion_feature(sa_text_features, img_features, attention_mask)
                    sa_output = self.sa_linear1(
                        self.dropout(torch.cat([sa_text_features, sa_fusion_features], dim=-1)))
                    sa_output = self.sa_linear2(sa_output)
                    sa_output = self.softmax(sa_output)
                return sa_output
            else:
                if self.args.add_pos:
                    att_opinion_features = self.opinion_attention(opinion_features, sa_text_features, pos_ids)
                    sa_fusion_features = self.sa_fusion_feature(sa_text_features, img_features, attention_mask)
                    sa_output = self.sa_linear1(self.dropout(torch.cat([sa_text_features, sa_fusion_features, att_opinion_features,pos_features], dim=-1)))
                    sa_output = self.sa_linear2(sa_output)
                    sa_output = self.softmax(sa_output)
                else:
                    att_opinion_features = self.opinion_attention(opinion_features, sa_text_features, pos_ids)
                    sa_fusion_features = self.sa_fusion_feature(sa_text_features, img_features, attention_mask)
                    sa_output = self.sa_linear1(self.dropout(
                        torch.cat([sa_text_features, sa_fusion_features, att_opinion_features], dim=-1)))
                    sa_output = self.sa_linear2(sa_output)
                    sa_output = self.softmax(sa_output)
                return sa_output

        sa_output=torch.tensor([0,0.33,0.33,0.33])
        sa_output=sa_output.unsqueeze(0).repeat(self.args.max_seq_len,1).unsqueeze(0).repeat(len(sa_text_features),1,1)
        for i in range(self.layer_num):
            sa_text_features=torch.cat([self.sa_linear3_stack[i](sa_text_features),sa_output],dim=-1)
            att_opinion_features = self.opinion_attention_stack[i](opinion_features, sa_text_features, pos_ids)
            sa_fusion_features = self.sa_fusion_feature_stack[i](sa_text_features, img_features, attention_mask)
            sa_output = self.sa_linear1_stack[i](self.dropout(torch.cat([sa_text_features, att_opinion_features, sa_fusion_features,pos_features], dim=-1)))
            sa_output = self.sa_linear2_stack[i](sa_output)
            sa_output = self.softmax(sa_output)
        return sa_output



class UMAS(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(UMAS, self).__init__()
        self.args=args
        if args.add_pos:
            self.pos_embedding = POS_embedding(args, args.pos_emb_dim)
        self.lstm = BiLSTM(args, pretrained_word_matrix).float()
        self.sa_lstm = BiLSTM(args, pretrained_word_matrix).float()
        # self.sa_lstm = nn.LSTM(input_size=args.ner_hidden_dim,hidden_size=args.sa_hidden_dim // 2,  bidirectional=True,batch_first=True)
        # self.op_lstm = BiLSTM(args, pretrained_word_matrix).float()

        self.ner_text_linear = nn.Linear(args.ner_hidden_dim, args.ner_hidden_dim)
        self.sa_text_linear = nn.Linear(args.ner_hidden_dim, args.sa_hidden_dim)
        self.opinion_text_linear = nn.Linear(args.ner_hidden_dim, args.sa_hidden_dim)

        self.fusion_linear1=nn.Linear(args.ner_hidden_dim,args.ner_hidden_dim)
        self.fusion_linear2=nn.Linear(args.ner_hidden_dim,args.ner_hidden_dim)
        self.gate_linear=nn.Linear(args.ner_hidden_dim*2,1)

        self.ner2op_linear=nn.Linear(args.ner_hidden_dim, args.sa_hidden_dim)

        # self.cat_linear=nn.Linear(args.ner_hidden_dim*2, args.sa_hidden_dim)


        # Transform each img vector as same dimensions ad the text vector
        self.dim_match1 = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.ner_hidden_dim),
            nn.Tanh()
        )
        self.dim_match2 = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.sa_hidden_dim),
            nn.Tanh()
        )

        self.ner_fusion_feature = Ner_Fusion_Feature(args)
        if self.args.add_pos:
            self.ner_linear = nn.Linear(args.ner_hidden_dim * 2 + args.pos_emb_dim, args.entity_num)
        else:
            self.ner_linear = nn.Linear(args.ner_hidden_dim * 2, args.entity_num)
        self.sa_stack=Sa_Stack(args,layer_num=self.args.sa_layer)

        if args.no_visual:
            if self.args.add_pos:
                self.ner_linear = nn.Linear(args.ner_hidden_dim + args.pos_emb_dim, args.entity_num)
            else:
                self.ner_linear = nn.Linear(args.ner_hidden_dim, args.entity_num)



    def forward(self, word_ids,pos_ids,char_ids, img_feature):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param pos_ids: (batch_size, max_seq_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """

        ner_output, sa_output=None,None
        # 文本特征表示
        text_features = self.lstm(word_ids, char_ids)
        if self.args.add_pos:
            pos_features = self.pos_embedding(pos_ids)
        else:
            pos_features=None
        if self.args.ner:
            ner_text_features = self.ner_text_linear(text_features)
            if self.args.no_visual:
                if self.args.add_pos:
                    ner_output = self.ner_linear(torch.cat([ner_text_features, pos_features], dim=-1))
                else:
                    ner_output = self.ner_linear(ner_text_features)
            else:
                img_features1 = self.dim_match1(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
                assert ner_text_features.size(-1) == img_features1.size(-1)
                ner_fusion_features = self.ner_fusion_feature(ner_text_features, img_features1)
                if self.args.add_pos:
                    ner_output=self.ner_linear(torch.cat([ner_text_features, ner_fusion_features,pos_features], dim=-1))
                else:
                    ner_output=self.ner_linear(torch.cat([ner_text_features, ner_fusion_features], dim=-1))


        if self.args.sa:
            sa_text_features= self.sa_lstm(word_ids, char_ids)
            opinion_features=self.opinion_text_linear(sa_text_features)

            new_sa=self.fusion_linear1(sa_text_features)
            new_t=self.fusion_linear2(text_features)
            gate=torch.sigmoid(self.gate_linear(torch.cat([new_sa,new_t],dim=-1))).repeat(1,1,self.args.ner_hidden_dim)
            new_sa_text=torch.mul(gate,sa_text_features)+torch.mul(1-gate,text_features)
            sa_text_features=torch.tanh(self.sa_text_linear(new_sa_text))


            if self.args.no_visual:
                sa_output=self.sa_stack(sa_text_features,opinion_features,pos_ids,None,pos_features)
            else:
                img_features2 = self.dim_match2(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
                assert sa_text_features.size(-1) == img_features2.size(-1)
                sa_output=self.sa_stack(sa_text_features,opinion_features,pos_ids,img_features2,pos_features)

        return ner_output,sa_output
