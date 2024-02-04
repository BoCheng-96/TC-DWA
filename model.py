from transformers import BertPreTrainedModel, BertModel, BertConfig, AutoConfig, AutoModelForMaskedLM, AutoModel
# from transformers.modeling_bert import BertOnlyMLMHead
from torch import nn
import sys
import torch
from torch.nn import Parameter
from isda import EstimatorCV
import torch.nn.functional as F

class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''
    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        # self.dense = nn.Linear(768, feature_num)
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        # x = self.fc(self.dense(x))
        return x

class TCDWAModel(nn.Module):
    def __init__(self, pretrained_lm, class_num, is_dual, beta):
        super(TCDWAModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_lm, output_attentions=True, output_hidden_states=True,
        #                                     output_norms=True)
        # self.bert = BertModel.from_pretrained(pretrained_lm)
        self.bert = AutoModel.from_pretrained(pretrained_lm)
        # self.bert = bert.bert
        # self.bert = BertModel(config, add_pooling_layer=False)
        # self.cls = BertOnlyMLMHead(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.75)
        self.dense = nn.Linear(768, 128)
        self.activation = nn.Tanh()
        if is_dual:
            self.estimator = EstimatorCV(768, class_num, beta)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.init_weights()
        # self.loss = nn.CrossEntropyLoss()
        # MLM head is not trained
        # for param in self.cls.parameters():
        #     param.requires_grad = False

    def constuct_features(self, remained_indices, hidden_states):
        # remained_indices: (batch_size, k)
        # hidden_states: (batch_size, seq_length, feature_dims)
        selected_token_features = torch.cat((hidden_states[0][0].unsqueeze(0), hidden_states[0][remained_indices[0]]), dim=0).unsqueeze(0)
        for i in range(1, hidden_states.shape[0]):
            one_ = torch.cat((hidden_states[i][0].unsqueeze(0), hidden_states[i][remained_indices[i]]), dim=0).unsqueeze(0)
            selected_token_features = torch.cat((selected_token_features, one_), dim=0)

        return selected_token_features

    def get_batch_cos_dis(self, cls_features, hidden_states):

        attention_distribution = []

        for i in range(hidden_states.shape[0]):
            similarity_distribution = []
            for j in range(hidden_states.shape[1]):
                attention_score = torch.cosine_similarity(cls_features[i].view(1,-1), hidden_states[i][j].view(1, -1))
                similarity_distribution.append(attention_score)
            similarity_distribution = torch.Tensor(similarity_distribution)
            norm_similarity_distribution = similarity_distribution / torch.sum(similarity_distribution, 0)
            attention_distribution.append(norm_similarity_distribution.reshape(1,-1))

        attention_distribution = torch.cat(attention_distribution, dim=0)

        return attention_distribution

    def select_tokens(self, input_ids, selected_indices, selected_weights, k, cls_id, sep_id):
        # selected_indices: (batch_size, k+2)
        # input_ids: (batch_size, seq_length)
        selected_input_ids = [input_ids[i][selected_indices[i]] for i in range(selected_indices.shape[0])]
        # (batch_size, k+2)
        selected_input_ids = torch.cat(selected_input_ids, dim=0).reshape(-1, k+2)
        # Remove [CLS] token and [SEP] token if have
        mask_matrix = (selected_input_ids != cls_id) & (selected_input_ids != sep_id)
        # (batch_size, k)
        remained_input_ids = [selected_input_ids[i][mask_matrix[i]][:k] for i in range(selected_input_ids.shape[0])]
        remained_input_ids = torch.cat(remained_input_ids, dim=0).reshape(-1, k)

        remained_indices = [selected_indices[i][mask_matrix[i]][:k] for i in range(selected_indices.shape[0])]
        # (batch_size, k)
        remained_indices = torch.cat(remained_indices, dim=0).reshape(-1, k)

        remained_weights = [selected_weights[i][mask_matrix[i]][:k] for i in range(selected_weights.shape[0])]
        # (batch_size, k)
        remained_weights = torch.cat(remained_weights, dim=0).reshape(-1, k)

        return remained_indices, remained_input_ids, remained_weights

    def forward(self, input_ids, similarity_type, cls_id, sep_id, shuffle_attentions=False, attention_threshold=0.8, k=5, layer_index=12, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        """
        Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.

        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        last_hidden_states, pooler_output, attentions, hidden_states, shuffled_hidden_states, norms = self.bert(input_ids,
                                                                                        selected_layer_index=layer_index,
                                                                                        attention_mask=attention_mask,
                                                                                        token_type_ids=token_type_ids,
                                                                                        position_ids=position_ids,
                                                                                        head_mask=head_mask,
                                                                                        inputs_embeds=inputs_embeds,
                                                                                        output_hidden_states=False,
                                                                                        output_attentions=True,
                                                                                        output_norms=True,
                                                                                        shuffle_attentions=shuffle_attentions,
                                                                                        attention_threshold=attention_threshold,
                                                                                        return_dict=False,)
        # Use hidden_states in layer_index to represent texts, here we try 7,9,12
        # Each element is a tuple which consists of 3 elements: ||f(x)||, ||αf(x)||, and ||Σαf(x)||.
        # shape of afx_norm (||αf(x)||) is (batch, num_heads, seq_length, seq_length)
        afx_norm = norms[1]
        # mean on 12 heads: (batch_size, seq_length, seq_length)
        mean_afx_norm = afx_norm.mean(1)
        # weights of each token for [CLS] token: (batch_size, seq_length)
        afx_norm_weights = mean_afx_norm[:, 0, :]
        # print(afx_norm_weights[0])
        # k: the number of selected anchor words
        # k = int(ratio * afx_norm_weights.shape[1])
        # print('{} anchor words are selected to represent texts'.format(k))
        # Use only [CLS] token
        if k==0:
            token_features = hidden_states[:, 0, :].unsqueeze(1)
            
            if shuffle_attentions:
                shuffled_token_features = shuffled_hidden_states[:, 0, :].unsqueeze(1)
            else:
                shuffled_token_features = None
            
            token_weights = torch.ones(hidden_states.shape[0], 1).cuda()
            
            remained_input_ids = torch.full([hidden_states.shape[0], 1], cls_id, dtype=torch.int64).cuda()
        else:
            # print('Selecting tokens with big weights!')
            if similarity_type == 'cos':
                cosine_weights = self.get_batch_cos_dis(hidden_states[:,0,:], hidden_states).cuda()
                # sort cosine weights
                sorted_weights, indices = torch.sort(cosine_weights, descending=True)
            elif similarity_type == 'norm':
                # sort afx_norm_weights
                sorted_weights, indices = torch.sort(afx_norm_weights, descending=True)
            else:
                print('No such type!')
            # select k+2 tokens: (batch_size, k+2)
            selected_indices = indices[:, :k+2]
            selected_weights = sorted_weights[:, :k+2]
            # (batch_size, k)
            remained_indices, remained_input_ids, remained_weights = self.select_tokens(input_ids, selected_indices, selected_weights, k, cls_id, sep_id)
            # (batch_size, 1)
            cls_weights = torch.ones(remained_indices.shape[0], 1).cuda()
            # (batch_size, k+1)
            token_weights = torch.cat((cls_weights, remained_weights), dim=1)
            # row normalization
            token_weights = F.normalize(token_weights, p=1, dim=1)
            # (batch_size, k+1, feature_dims)
            token_features = self.constuct_features(remained_indices, hidden_states)
            
            if shuffle_attentions:
                shuffled_token_features = self.constuct_features(remained_indices, shuffled_hidden_states)
            else:
                shuffled_token_features = None

        mean_hidden_states = torch.mean(last_hidden_states, dim=1)
        # text_features = torch.matmul(selected_weights.unsqueeze(1), selected_token_features).squeeze(1)
        # print('Succeed in constructing texts with selected token features!')
        # selected_input_ids = torch.take(input_ids, indices)[:, :k]
        # mean_hidden_states = torch.mean(last_hidden_states[0], dim=1)
        # print('mean_hidden_states in BertClusteringModel: {} and shape = {}'.format(mean_hidden_states, mean_hidden_states.shape))
        # if pred_mode == "classification":
        #     trans_states = self.dense(selected_token_features)
        # #     # trans_states = self.activation(trans_states)
        # #     # trans_states = self.dropout(trans_states)
        # #     # (batch_size, k, l)
        # #     # logits = self.classifier(trans_states)
        # #     # print('logits in BertClusteringModel: {} and shape = {}'.format(logits, logits.shape))
        # # # elif pred_mode == "mlm":
        # # #     logits = self.cls(last_hidden_states)
        # else:
        #     sys.exit("Wrong pred_mode!")
        #
        #
        # return pooler_output, selected_token_features, trans_states, final_selected_weights, remained_input_ids
        return mean_hidden_states, pooler_output, token_features, shuffled_token_features, token_weights, remained_input_ids
