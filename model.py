import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class BiLSTM_Zhang_Attention(nn.Module):
    
    def __init__(self, embedding_dim, n_hidden, num_classes, model_type='LSTM', vocab_size=100, bidirectional=False):
        super().__init__()

        self.hidden_size = (n_hidden * 2 if bidirectional else n_hidden)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert any(n in model_type.split('-')[0] for n in ['LSTM', 'GRU'])
        self.model_type = model_type
        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        else:
            self.gru = nn.GRU(embedding_dim, self.hidden_size)
        self.word = nn.Linear(self.hidden_size, 1, bias=False)
        self.out = nn.Linear(self.hidden_size,
                             num_classes, bias=False)  # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix

    def attention_net(self, lstm_output):
        attn_weights = self.word(lstm_output).squeeze(2)
        soft_attn_weights = torch.sigmoid(attn_weights)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, lstm_output, human_attention):
        soft_attn_weights = human_attention
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X, human_attention=None):
        input = X.permute(1, 0, 2)  # change input tensor dimesion to : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)
        cell_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)

        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        else:
            output, final_hidden_state = self.gru(input, hidden_state)
        output = output.permute(1, 0, 2)  # change output tensor dimesion to: [batch_size, len_seq, n_hidden]

        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(output, human_attention)
        else:
            attn_output, alpha = self.attention_net(output)

        return self.out(attn_output), alpha
        
        
class BiLSTM_Bar_Attention(nn.Module):
    def __init__(self, embedding_dim, n_hidden, num_classes, whidden_size, evidence_size, hidden_layer_size, model_type='LSTM', vocab_size=100, bidirectional=False):
        super().__init__()

        self.hidden_size = (n_hidden * 2 if bidirectional else n_hidden)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert any(n in model_type.split('-')[0] for n in ['LSTM', 'GRU'])
        self.model_type = model_type
        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        else:
            self.gru = nn.GRU(embedding_dim, self.hidden_size)
        self.whidden_layer = nn.Linear(self.hidden_size, whidden_size)
        self.evidence_layer = nn.Linear(whidden_size, evidence_size)
        self.attention_layer = nn.Linear(evidence_size, 1)
        self.before_out = nn.Linear(whidden_size, hidden_layer_size)
        self.out = nn.Linear(hidden_layer_size, num_classes)  # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix

    def attention_net(self, rnn_output):
        attn_evidence = torch.tanh(self.evidence_layer(rnn_output))
        attn_weights = self.attention_layer(attn_evidence).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        sen_rep = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return sen_rep, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, rnn_output, human_attention):
        soft_attn_weights = human_attention
        sen_rep = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return sen_rep, soft_attn_weights

    def forward(self, X, human_attention=None):
        input = X.permute(1, 0, 2)  # change input tensor dimesion to : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)
        cell_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)

        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        else:
            output, final_hidden_state = self.gru(input, hidden_state)
        output = output.permute(1, 0, 2)  # change output tensor dimesion to: [batch_size, len_seq, n_hidden]
        output = torch.tanh(self.whidden_layer(output))
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(output, human_attention)
        else:
            attn_output, alpha = self.attention_net(output)
        before_output = torch.tanh(self.before_out(attn_output))

        return self.out(before_output), alpha


class BiLSTM_HUG_Attention(nn.Module):
    def __init__(self, embedding_dim, n_hidden, num_classes, model_type='LSTM', vocab_size=100, bidirectional=False):
        super().__init__()

        self.hidden_size = (n_hidden * 2 if bidirectional else n_hidden)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert any(n in model_type.split('-')[0] for n in ['LSTM', 'GRU'])
        self.model_type = model_type
        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        else:
            self.gru = nn.GRU(embedding_dim, self.hidden_size)
        self.before_out = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size,
                             num_classes)  # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        self.convert_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

    def attention_net(self, rnn_output, final_state):
        hidden = final_state.view(-1, self.hidden_size, 1)
        converted_output = torch.tanh(self.convert_layer(rnn_output))
        attn_weights = torch.bmm(converted_output, hidden).squeeze(2)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((hidden.squeeze(2), highlighted_context), dim=1))
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, rnn_output, final_state, human_attention):
        hidden = final_state.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((hidden.squeeze(2), highlighted_context), dim=1))
        return context, soft_attn_weights

    def forward(self, X, human_attention=None):
        input = X.permute(1, 0, 2)  # change input tensor dimesion to : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)
        cell_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)

        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        else:
            output, final_hidden_state = self.gru(input, hidden_state)
        output = output.permute(1, 0, 2)  # change output tensor dimesion to: [batch_size, len_seq, n_hidden]

        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(output, final_hidden_state, human_attention)
        else:
            attn_output, alpha = self.attention_net(output, final_hidden_state)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha
    
    
class BiLSTM_HUGA_Attention(nn.Module):
    def __init__(self, embedding_dim, n_hidden, num_classes, model_type='LSTM', vocab_size=100, bidirectional=False):
        super().__init__()

        self.hidden_size = (n_hidden * 2 if bidirectional else n_hidden)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert any(n in model_type.split('-')[0] for n in ['LSTM', 'GRU'])
        self.model_type = model_type
        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        else:
            self.gru = nn.GRU(embedding_dim, self.hidden_size)
        self.before_out = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size,
                             num_classes)  # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        self.dropout = nn.Dropout(p=0.2)

    def attention_net(self, rnn_output, final_state):
        hidden = final_state.view(-1, self.hidden_size, 1)
        attn_weights = torch.bmm(rnn_output, hidden).squeeze(2)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((hidden.squeeze(2), highlighted_context), dim=1))
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, rnn_output, final_state, human_attention):
        hidden = final_state.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((hidden.squeeze(2), highlighted_context), dim=1))
        return context, soft_attn_weights

    def forward(self, X, human_attention=None):
        input = X.permute(1, 0, 2)  # change input tensor dimesion to : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)
        cell_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)

        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        else:
            output, final_hidden_state = self.gru(input, hidden_state)
        output = output.permute(1, 0, 2)  # change output tensor dimesion to: [batch_size, len_seq, n_hidden]

        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(output, final_hidden_state, human_attention)
        else:
            attn_output, alpha = self.attention_net(output, final_hidden_state)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha
    

class BiLSTM_HUGS_Attention(nn.Module):
    def __init__(self, embedding_dim, n_hidden, num_classes, model_type='LSTM', vocab_size=100, bidirectional=False):
        super().__init__()

        self.hidden_size = (n_hidden * 2 if bidirectional else n_hidden)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert any(n in model_type.split('-')[0] for n in ['LSTM', 'GRU'])
        self.model_type = model_type
        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        else:
            self.gru = nn.GRU(embedding_dim, self.hidden_size)
        self.before_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size,
                             num_classes)  # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        self.convert_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

    def attention_net(self, rnn_output, final_state):
        hidden = final_state.view(-1, self.hidden_size, 1)
        converted_output = torch.tanh(self.convert_layer(rnn_output))
        attn_weights = torch.bmm(converted_output, hidden).squeeze(2)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(hidden.squeeze(2))
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, rnn_output, final_state, human_attention):
        hidden = final_state.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(hidden.squeeze(2))
        return context, soft_attn_weights

    def forward(self, X, human_attention=None):
        input = X.permute(1, 0, 2)  # change input tensor dimesion to : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)
        cell_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)

        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        else:
            output, final_hidden_state = self.gru(input, hidden_state)
        output = output.permute(1, 0, 2)  # change output tensor dimesion to: [batch_size, len_seq, n_hidden]

        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(output, final_hidden_state, human_attention)
        else:
            attn_output, alpha = self.attention_net(output, final_hidden_state)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha
        
        
class BiLSTM_HUGW_Attention(nn.Module):
    def __init__(self, embedding_dim, n_hidden, num_classes, model_type='LSTM', vocab_size=100, bidirectional=False):
        super().__init__()

        self.hidden_size = (n_hidden * 2 if bidirectional else n_hidden)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert any(n in model_type.split('-')[0] for n in ['LSTM', 'GRU'])
        self.model_type = model_type
        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size)
        else:
            self.gru = nn.GRU(embedding_dim, self.hidden_size)
        self.before_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size,
                             num_classes)  # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        self.convert_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

    def attention_net(self, rnn_output, final_state):
        hidden = final_state.view(-1, self.hidden_size, 1)
        converted_output = torch.tanh(self.convert_layer(rnn_output))
        attn_weights = torch.bmm(converted_output, hidden).squeeze(2)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(highlighted_context)
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, rnn_output, final_state, human_attention):
        hidden = final_state.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(rnn_output.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(highlighted_context)
        return context, soft_attn_weights

    def forward(self, X, human_attention=None):
        input = X.permute(1, 0, 2)  # change input tensor dimesion to : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)
        cell_state = Variable(torch.zeros(1, len(X), self.hidden_size)).to(input.device)

        if self.model_type.split('-')[0] in ['LSTM', 'BiLSTM']:
            output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        else:
            output, final_hidden_state = self.gru(input, hidden_state)
        output = output.permute(1, 0, 2)  # change output tensor dimesion to: [batch_size, len_seq, n_hidden]

        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(output, final_hidden_state, human_attention)
        else:
            attn_output, alpha = self.attention_net(output, final_hidden_state)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha


class BertHUGAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.before_out = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_labels)
        self.convert_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        self.init_weights()
    
    def attention_net(self, bert_last_hidden_state, bert_pooled_output):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1) # hidden: (batch_size, hidden_size, 1)
        converted_last_hidden_state = torch.tanh(self.convert_layer(bert_last_hidden_state)) # converted_last_hidden_state
        attn_weights = torch.bmm(converted_last_hidden_state, hidden).squeeze(2) # attn_weights: (batch_size, sequence_length)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((bert_pooled_output, highlighted_context), dim=1))
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, bert_last_hidden_state, bert_pooled_output, human_attention):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((bert_pooled_output, highlighted_context), dim=1))
        return context, soft_attn_weights

    def forward(self, X, mask, human_attention=None):
        '''
        last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        Sequence of hidden-states at the output of the last layer of the model.

        pooler_output: (torch.FloatTensor: of shape (batch_size, hidden_size))
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function.
        The Linear layer weights are trained from the next sentence prediction
        (classification) objective during pre-training.

        human_attention: (torch.FloatTensor of shape (batch_size, sequence_length))
        '''
        last_hidden_state, pooled_output = self.bert(X, token_type_ids=None, attention_mask=mask)
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(last_hidden_state, pooled_output, human_attention)
        else:
            attn_output, alpha = self.attention_net(last_hidden_state, pooled_output)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha


class BertHUGAAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.before_out = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=0.2)

        self.init_weights()
    
    def attention_net(self, bert_last_hidden_state, bert_pooled_output):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1) # hidden: (batch_size, hidden_size, 1)
        attn_weights = torch.bmm(bert_last_hidden_state, hidden).squeeze(2) # attn_weights: (batch_size, sequence_length)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((bert_pooled_output, highlighted_context), dim=1))
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, bert_last_hidden_state, bert_pooled_output, human_attention):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(torch.cat((bert_pooled_output, highlighted_context), dim=1))
        return context, soft_attn_weights

    def forward(self, X, mask, human_attention=None):
        '''
        last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        Sequence of hidden-states at the output of the last layer of the model.

        pooler_output: (torch.FloatTensor: of shape (batch_size, hidden_size))
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function.
        The Linear layer weights are trained from the next sentence prediction
        (classification) objective during pre-training.

        human_attention: (torch.FloatTensor of shape (batch_size, sequence_length))
        '''
        last_hidden_state, pooled_output = self.bert(X, token_type_ids=None, attention_mask=mask)
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(last_hidden_state, pooled_output, human_attention)
        else:
            attn_output, alpha = self.attention_net(last_hidden_state, pooled_output)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha


class BertHUGSAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.before_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_labels)
        self.convert_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        self.init_weights()
    
    def attention_net(self, bert_last_hidden_state, bert_pooled_output):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1) # hidden: (batch_size, hidden_size, 1)
        converted_last_hidden_state = torch.tanh(self.convert_layer(bert_last_hidden_state)) # converted_last_hidden_state
        attn_weights = torch.bmm(converted_last_hidden_state, hidden).squeeze(2) # attn_weights: (batch_size, sequence_length)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(bert_pooled_output)
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, bert_last_hidden_state, bert_pooled_output, human_attention):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(bert_pooled_output)
        return context, soft_attn_weights

    def forward(self, X, mask, human_attention=None):
        '''
        last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        Sequence of hidden-states at the output of the last layer of the model.

        pooler_output: (torch.FloatTensor: of shape (batch_size, hidden_size))
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function.
        The Linear layer weights are trained from the next sentence prediction
        (classification) objective during pre-training.

        human_attention: (torch.FloatTensor of shape (batch_size, sequence_length))
        '''
        last_hidden_state, pooled_output = self.bert(X, token_type_ids=None, attention_mask=mask)
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(last_hidden_state, pooled_output, human_attention)
        else:
            attn_output, alpha = self.attention_net(last_hidden_state, pooled_output)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha


class BertHUGWAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.before_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_labels)
        self.convert_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        self.init_weights()
    
    def attention_net(self, bert_last_hidden_state, bert_pooled_output):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1) # hidden: (batch_size, hidden_size, 1)
        converted_last_hidden_state = torch.tanh(self.convert_layer(bert_last_hidden_state)) # converted_last_hidden_state
        attn_weights = torch.bmm(converted_last_hidden_state, hidden).squeeze(2) # attn_weights: (batch_size, sequence_length)
        soft_attn_weights = torch.sigmoid(attn_weights)
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(highlighted_context)
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, bert_last_hidden_state, bert_pooled_output, human_attention):
        hidden = bert_pooled_output.view(-1, self.hidden_size, 1)
        soft_attn_weights = human_attention
        highlighted_context = torch.bmm(bert_last_hidden_state.transpose(1, 2),
        soft_attn_weights.unsqueeze(2)).squeeze(2)
        context = torch.tanh(highlighted_context)
        return context, soft_attn_weights

    def forward(self, X, mask, human_attention=None):
        '''
        last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        Sequence of hidden-states at the output of the last layer of the model.

        pooler_output: (torch.FloatTensor: of shape (batch_size, hidden_size))
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function.
        The Linear layer weights are trained from the next sentence prediction
        (classification) objective during pre-training.

        human_attention: (torch.FloatTensor of shape (batch_size, sequence_length))
        '''
        last_hidden_state, pooled_output = self.bert(X, token_type_ids=None, attention_mask=mask)
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(last_hidden_state, pooled_output, human_attention)
        else:
            attn_output, alpha = self.attention_net(last_hidden_state, pooled_output)
        before_output = torch.tanh(self.before_out(attn_output))
        before_output = self.dropout(before_output)
        return self.out(before_output), alpha

        
class BertBarAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.whidden_size = 200
        self.evidence_size = 100
        self.hidden_layer_size = 50
        
        self.whidden_layer = nn.Linear(self.hidden_size, self.whidden_size)
        self.evidence_layer = nn.Linear(self.whidden_size, self.evidence_size)
        self.attention_layer = nn.Linear(self.evidence_size, 1)
        self.before_out = nn.Linear(self.whidden_size, self.hidden_layer_size)
        self.out = nn.Linear(self.hidden_layer_size, self.num_labels)

        self.init_weights()
    
    def attention_net(self, bert_last_hidden_state):
        attn_evidence = torch.tanh(self.evidence_layer(bert_last_hidden_state))
        attn_weights = self.attention_layer(attn_evidence).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        sen_rep = torch.bmm(bert_last_hidden_state.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return sen_rep, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, bert_last_hidden_state, human_attention):
        soft_attn_weights = human_attention
        sen_rep = torch.bmm(bert_last_hidden_state.transpose(1, 2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return sen_rep, soft_attn_weights

    def forward(self, X, mask, human_attention=None):
        '''
        last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        Sequence of hidden-states at the output of the last layer of the model.

        pooler_output: (torch.FloatTensor: of shape (batch_size, hidden_size))
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function.
        The Linear layer weights are trained from the next sentence prediction
        (classification) objective during pre-training.

        human_attention: (torch.FloatTensor of shape (batch_size, sequence_length))
        '''
        last_hidden_state, pooled_output = self.bert(X, token_type_ids=None, attention_mask=mask)
        last_hidden_state = torch.tanh(self.whidden_layer(last_hidden_state))
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(last_hidden_state, human_attention)
        else:
            attn_output, alpha = self.attention_net(last_hidden_state)
        before_output = torch.tanh(self.before_out(attn_output))
        return self.out(before_output), alpha
        
        
class BertZhangAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.word = nn.Linear(self.hidden_size, 1, bias=False)
        self.out = nn.Linear(self.hidden_size, self.num_labels, bias=False)
        self.dropout = nn.Dropout(p=0.2)

        self.init_weights()

    def attention_net(self, bert_last_hidden_state):
        attn_weights = self.word(bert_last_hidden_state).squeeze(2)
        soft_attn_weights = torch.sigmoid(attn_weights)
        context = torch.bmm(bert_last_hidden_state.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights  # .data.numpy()

    def human_attention_net(self, bert_last_hidden_state, human_attention):
        soft_attn_weights = human_attention
        context = torch.bmm(bert_last_hidden_state.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X, mask, human_attention=None):
        '''
        last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        Sequence of hidden-states at the output of the last layer of the model.

        pooler_output: (torch.FloatTensor: of shape (batch_size, hidden_size))
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function.
        The Linear layer weights are trained from the next sentence prediction
        (classification) objective during pre-training.

        human_attention: (torch.FloatTensor of shape (batch_size, sequence_length))
        '''
        last_hidden_state, pooled_output = self.bert(X, token_type_ids=None, attention_mask=mask)
        if human_attention is not None:
            attn_output, alpha = self.human_attention_net(last_hidden_state, human_attention)
        else:
            attn_output, alpha = self.attention_net(last_hidden_state)

        return self.out(attn_output), alpha