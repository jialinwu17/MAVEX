# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tensorflow as tf
import copy
import json, pickle
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from .utils import PreTrainedModel
import pdb
from torch.nn.utils.weight_norm import weight_norm

from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = GeLU()

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)


        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims, dropout):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(LayerNorm(out_dim))
            layers.append(GeLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if dims[-1] != 1:
            layers.append(LayerNorm(dims[-1]))
        layers.append(GeLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class MHAtt(nn.Module):
    def __init__(self, num_head, num_hid, dropout):
        super(MHAtt, self).__init__()
        self.num_hid = num_hid
        self.num_head = num_head
        self.linear_v = nn.Linear(num_hid, num_hid)
        self.linear_k = nn.Linear(num_hid, num_hid)
        self.linear_q = nn.Linear(num_hid, num_hid)
        self.linear_merge = nn.Linear(num_hid, num_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_head,
            int(self.num_hid // self.num_head)
        ).transpose(1, 2)  # batch, n_items, num_head, num_feats --> batch, num_head, n_items, num_feats

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_head,
            int(self.num_hid // self.num_head)
        ).transpose(1, 2)  # batch, n_items, num_head, num_feats --> batch, num_head, n_items, num_feats

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_head,
            int(self.num_hid // self.num_head)
        ).transpose(1, 2)  # batch, n_items, num_head, num_feats --> batch, num_head, n_items, num_feats
        atted, att_map = self.att(v, k, q,
                                  mask)  # batch, num_head, q_items, num_feats |||batch, num_head, q_items, k_items
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.num_hid
        )  # batch, q_items, num_head * num_feats
        atted = self.linear_merge(atted)

        return atted, att_map

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
            # batch, num_head, n_items, num_feats ** batch, num_head, num_feats, n_items,
        ) / math.sqrt(d_k)  # batch, num_head, n_items, n_items

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        if query.size(1) > 1:
            att_map = self.dropout(att_map)

        return torch.matmul(att_map, value), att_map


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, num_hid, drop_out):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=num_hid,
            mid_size=num_hid,
            out_size=num_hid,
            dropout_r=drop_out,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------


class SGASimple(nn.Module):
    def __init__(self, num_head, num_hid, num_out, dropout, input_dim1=None, input_dim2=None):
        super(SGASimple, self).__init__()
        self.input1 = None
        if input_dim1 is not None:
            self.input1 = FC(input_dim1, num_hid, dropout_r=dropout, use_relu=True)
        self.input2 = None
        if input_dim2 is not None:
            self.input2 = FC(input_dim2, num_hid, dropout_r=dropout, use_relu=True)

        self.mhatt2 = MHAtt(num_head, num_hid, dropout)
        self.ffn = FFN(num_hid, dropout)
        self.out = FC(num_hid, num_out)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(num_hid)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(num_hid)

    def forward(self, x, y, x_mask=None, y_mask=None):
        if self.input1 is not None:
            x = self.input1(x)
        if self.input2 is not None:
            y = self.input2(y)
        mhatt2, att_map = self.mhatt2(y, y, x, y_mask)
        x = self.norm2(x + self.dropout2(mhatt2))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
        x = self.out(x)
        return x


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
            self,
            vocab_size_or_config_json_file,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            v_feature_size=2048,
            v_target_size=1601,
            v_hidden_size=768,
            v_num_hidden_layers=3,
            v_num_attention_heads=12,
            v_intermediate_size=3072,
            bi_hidden_size=1024,
            bi_num_attention_heads=16,
            v_attention_probs_dropout_prob=0.1,
            v_hidden_act="gelu",
            v_hidden_dropout_prob=0.1,
            v_initializer_range=0.2,
            v_biattention_id=[0, 1],
            t_biattention_id=[10, 11],
            visual_target=0,
            fast_mode=False,
            fixed_v_layer=0,
            fixed_t_layer=0,
            in_batch_pairs=False,
            fusion_method="mul",
            dynamic_attention=False,
            with_coattention=True,
            objective=0,
            num_negative=128,
            model="bert",
            task_specific_tokens=False,
            visualization=False,
    ):

        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        assert len(v_biattention_id) == len(t_biattention_id)
        assert max(v_biattention_id) < v_num_hidden_layers
        assert max(t_biattention_id) < num_hidden_layers

        if isinstance(vocab_size_or_config_json_file, str) or (
                sys.version_info[0] == 2
                and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.v_feature_size = v_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_hidden_layers = v_num_hidden_layers
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_biattention_id = v_biattention_id
            self.t_biattention_id = t_biattention_id
            self.v_target_size = v_target_size
            self.bi_hidden_size = bi_hidden_size
            self.bi_num_attention_heads = bi_num_attention_heads
            self.visual_target = visual_target
            self.fast_mode = fast_mode
            self.fixed_v_layer = fixed_v_layer
            self.fixed_t_layer = fixed_t_layer

            self.model = model
            self.in_batch_pairs = in_batch_pairs
            self.fusion_method = fusion_method
            self.dynamic_attention = dynamic_attention
            self.with_coattention = with_coattention
            self.objective = objective
            self.num_negative = num_negative
            self.task_specific_tokens = task_specific_tokens
            self.visualization = visualization
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.task_specific_tokens = config.task_specific_tokens
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.task_specific_tokens:
            self.task_embeddings = nn.Embedding(20, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, task_ids=None, position_ids=None):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.task_specific_tokens:
            task_embeddings = self.task_embeddings(task_ids)
            embeddings = torch.cat(
                [embeddings[:, 0:1], task_embeddings, embeddings[:, 1:]], dim=1
            )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(
                self.padding_idx + 1,
                seq_length + self.padding_idx + 1,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.visualization = config.visualization

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )

        self.visualization = config.visualization

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)

            # given pool embedding, Linear and Sigmoid layer.
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))

            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):
    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding, txt_attention_mask):
        self_output, attention_probs = self.self(
            input_tensor, attention_mask, txt_embedding, txt_attention_mask
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.v_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):
    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, txt_embedding, txt_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask=None,
            use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask=None,
            use_co_attention_mask=False,
    ):
        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )
        self.v_layer = nn.ModuleList(
            [copy.deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)]
        )
        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))]
        )

    def forward(
            self,
            txt_embedding,
            image_embedding,
            txt_attention_mask,
            txt_attention_mask2,
            image_attention_mask,
            co_attention_mask=None,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0)
                        .expand(batch_size, batch_size, num_regions, v_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0)
                        .expand(batch_size, batch_size, 1, 1, num_regions)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                        .expand(batch_size, batch_size, num_words, t_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, 1, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_words)
                )
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, num_regions, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, num_regions, num_words)
                )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[
                    count
                ](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding,
                image_attention_mask,
                txt_embedding,
                txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask
            )

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )


class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(
            self, sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
    ):

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v = self.imagePredictions(sequence_output_v)

        return prediction_scores_t, prediction_scores_v, seq_relationship_score


class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # torch.nn.init.kaiming_normal_(module.weight.data, a=0.001)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        # initilize word embedding
        if config.model == "bert":
            self.embeddings = BertEmbeddings(config)
        elif config.model == "roberta":
            self.embeddings = RobertaEmbeddings(config)

        self.task_specific_tokens = config.task_specific_tokens

        # initlize the vision embedding
        self.v_embeddings = BertImageEmbeddings(config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.apply(self.init_weights)

    def forward(
            self,
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            co_attention_mask=None,
            task_ids=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask2 = attention_mask.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask2 = extended_attention_mask2.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_txt.size(0), input_imgs.size(1), input_txt.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_attention_mask2,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return (
            encoded_layers_t,
            encoded_layers_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        )


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        # TODO: we want to make the padding_idx == 0, however, with custom initilization, it seems it will have a bias.
        # Let's do masking for now
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        # embeddings = self.LayerNorm(img_embeddings+loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


from torch.nn import Linear
import torch.nn.functional as F

class ImageSegmentEncoder(nn.Module):
    def __init__(self, hidden_dim, output_dim=1024):
        super(ImageSegmentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = 4
        self.feat_dim = 1024

        self.num_search_per_segment = 3
        self.num_head = 4
        self.max_segment = 4

        self.output_dim = output_dim
        self.attentive_pooling = AttentivePooling(768, hidden_dim)
        self.qid2knowledge = pickle.load(open('qid2knowledge_cleaned_mavex.pkl', 'rb'))
        self.qid2q = pickle.load(open('qid2q.pkl', 'rb'))
        self.small_bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")

        self.sga_q = SGASimple(self.num_head, hidden_dim, output_dim, dropout=0.2, input_dim1=hidden_dim, input_dim2=1024)
        self.sga_default = SGASimple(self.num_head, hidden_dim, output_dim, dropout=0.2, input_dim1=768, input_dim2=1024)
        self.sga_mavex = SGASimple(self.num_head, hidden_dim, output_dim, dropout=0.2, input_dim1=1024, input_dim2=2048)
        self.max_segment = 5
        self.hidden_dim = hidden_dim
        if not os.path.isfile('segment2span.pkl'):
            self.qid2seg_spans = {}
            qids = self.qid2q.keys()
            for qid in qids:
                _ = self.segment2span(qid)
            pickle.dump(self.qid2seg_spans, open('segment2span.pkl', 'wb'))
        else:
            self.qid2seg_spans = pickle.load( open('segment2span.pkl', 'rb'))

    def segment2span(self, question_id, max_question_len=22):
        if question_id in self.qid2seg_spans:
            return self.qid2seg_spans[question_id]
        segment_list = self.qid2knowledge[int(question_id)]['segment_list']
        question = self.qid2q[int(question_id)]
        question_tokens = self.small_bert_tokenizer.convert_tokens_to_ids(
            self.small_bert_tokenizer.tokenize(question))[:max_question_len]
        segments_spans = {}
        for segment in segment_list:
            segment_tokens = self.small_bert_tokenizer.convert_tokens_to_ids(
                self.small_bert_tokenizer.tokenize(segment))
            found = -1
            for i in range(len(question_tokens) - len(segment_tokens) + 1):
                if question_tokens[i:i + len(segment_tokens)] == segment_tokens:
                    found = i
            if found >= 0:
                segments_spans[segment] = [found, found + len(segment_tokens)]
            else:
                segments_spans[segment] = [-1, -1]
        self.qid2seg_spans[question_id] = segments_spans
        return segments_spans

    def forward(self, textual_features, visual_features, pooled_features, image_features, question_ids):
        # output default batch 768
        # output segments =
        batch_size = len(question_ids)
        batch_idx = []
        output_segment_spans = []
        output_value_gather_idx = []
        output_q_gather_idx = []
        num_objs = visual_features.size(1)

        for i in range(batch_size):
            segment_spans = self.segment2span(int(question_ids[i]))
            segment_list = list(segment_spans.keys())
            segment2image_knowledge = self.qid2knowledge[int(question_ids[i])]['segment2image_knowledge']
            assert '||default||' in segment_list[0]
            num_seg = 0
            q_gather_idx = [9999] * self.max_segment

            for segment in segment_list:
                if ('||default||' not in segment) and ('MAVEx' not in segment):
                    output_segment_spans.append(self.segment2span(question_ids[i])[segment])
                    linked_objs = segment2image_knowledge[segment]
                    ct = 0

                    value_gather_idx = [batch_size * num_objs] * self.num_search_per_segment
                    for obj in linked_objs:
                        if ct == self.num_search_per_segment:
                            break
                        obj_id = int(obj.split('||')[0])
                        value_gather_idx[ct] = i * num_objs + obj_id # visual_features[i, obj_id] #
                        ct += 1
                    q_gather_idx[num_seg] = len(batch_idx)
                    output_value_gather_idx.extend(value_gather_idx)
                    batch_idx.append(i)
                    num_seg += 1
                if num_seg == self.max_segment:
                    break
            output_q_gather_idx.extend(q_gather_idx)
        output_q_gather_idx_real = []
        for _ in output_q_gather_idx:
            if _ == 9999:
                output_q_gather_idx_real.append(len(batch_idx))
            else:
                output_q_gather_idx_real.append(_)
        tmp_visual_features = torch.cat([visual_features.view(batch_size * num_objs, -1), torch.zeros((1, visual_features.size(-1))).cuda()])
        output_values = tmp_visual_features[output_value_gather_idx].view(len(batch_idx), self.num_search_per_segment, -1)
        output_key = self.attentive_pooling(textual_features, output_segment_spans, batch_idx).unsqueeze(1)
        output_default_key = textual_features[:, 0:1]
        output_default_values = visual_features[:, 0: 1]

        output_q_ = self.sga_q(output_key, output_values).squeeze(1)
        output_q_ = torch.cat([output_q_, torch.zeros((1, output_q_.size(-1))).cuda()])
        output_q = output_q_[output_q_gather_idx_real].view(batch_size, self.max_segment, -1)

        output_default = self.sga_default(output_default_key, output_default_values).view(batch_size, -1)
        output_mavex_key = pooled_features.view(batch_size, 1, 1, -1).repeat(1, 5, 1, 1).view(batch_size*5, 1, -1) # batch * 5,  1, dim
        output_mavex = self.sga_mavex(output_mavex_key, image_features.view(batch_size*5, 5, -1)).view(batch_size, 5, -1)
        return output_q, output_default, output_mavex, torch.ones((batch_size, 5)).float().cuda()


class SequenceEncoder(nn.Module):
    def __init__(self, use_search_word, bert_size, output_dim = 512, dropout=0.1):
        super(SequenceEncoder, self).__init__()
        self.text_bert = AutoModel.from_pretrained("prajjwal1/bert-%s" % bert_size)
        if bert_size == 'tiny':
            self.feat_dim = 128
        elif bert_size == 'mini':
            self.feat_dim = 256
        else:
            self.feat_dim = 512
        self.use_search_word = use_search_word
        self.output_dim = output_dim
        self.proj = FCNet([self.feat_dim, output_dim], dropout)

    def forward(self, seqs):
        # the last dim is seq length  seqs: [batch, num_sents]
        original_shape = list(seqs.size())
        flattened_seqs = seqs.reshape(-1, original_shape[-1])
        flattened_masks = (flattened_seqs.sum(1) > 0).long().detach().cpu().numpy()
        gather_idxs = []

        for i in range(len(flattened_masks)):
            if flattened_masks[i]:
                gather_idxs.append(i)

        if len(gather_idxs):
            valid_seqs = flattened_seqs[gather_idxs]
            processed_valid_seqs = self.proj(self.text_bert(valid_seqs).pooler_output)
            processed_valid_seqs = torch.cat([processed_valid_seqs, torch.zeros((1, processed_valid_seqs.size(1))).cuda()])
        else:
            processed_valid_seqs =  torch.zeros((1, self.feat_dim)).cuda()

        post_gather_idxs = []
        ct = 0
        for i in range(len(flattened_masks)):
            if flattened_masks[i]:
                post_gather_idxs.append(ct)
                ct += 1
            else:
                post_gather_idxs.append(0)
        assert ct == len(gather_idxs)
        processed_flattened_seqs = processed_valid_seqs[post_gather_idxs]
        processed_seqs = processed_flattened_seqs.reshape(original_shape[:-1] + [self.output_dim])
        return processed_seqs


class AttentivePooling(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(AttentivePooling, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.att_pool_fc1 = FCNet([self.feat_dim, 1], dropout=0.1)
        self.att_pool_feats = FCNet([self.feat_dim, hidden_dim], dropout=0.2)
        self.att_pool_feats_edge = FCNet([self.feat_dim, hidden_dim], dropout=0.2)
        self.att_pool_output = FCNet([hidden_dim * 2, hidden_dim], dropout=0.2)

    def forward(self, textual_features, segment_spans, batch_idx):
        batch_size, seq_len, _ = textual_features.size()
        exp_attention1 = torch.exp(self.att_pool_fc1(textual_features)).view(1, batch_size, seq_len, 1) # batch, 23, 1
        proj_feats = self.att_pool_feats(textual_features).view(1, batch_size, seq_len, -1)  # batch, 23, 768
        proj_feats_edge = self.att_pool_feats_edge(textual_features[:, 0])  # batch, 23, 768

        proj_feats_mask_up = np.zeros((len(segment_spans), batch_size, seq_len, 1))
        proj_feats_mask_down = np.zeros((len(segment_spans), batch_size, seq_len, 1))

        proj_feats_edge_mask = np.zeros((len(segment_spans), batch_size, 1))

        for i in range(len(segment_spans)):
            proj_feats_edge_mask[i, batch_idx[i]] = 1.
            if segment_spans[i][0] == -1:
                proj_feats_mask_up[i, batch_idx[i], 0] = 1.
                proj_feats_mask_down[i, batch_idx[i], 0] = 1.
            else:
                proj_feats_mask_up[i, batch_idx[i], segment_spans[i][0] + 1: segment_spans[i][1] + 1] = 1.
                proj_feats_mask_down[i, batch_idx[i], 1: ] = 1.
        proj_feats_edge_mask = torch.from_numpy(proj_feats_edge_mask).float().cuda()
        proj_feats_mask_down = torch.from_numpy(proj_feats_mask_down).float().cuda()
        proj_feats_mask_up = torch.from_numpy(proj_feats_mask_up).float().cuda()

        seg_feats_edge = (proj_feats_edge_mask * proj_feats_edge.view(1, batch_size, -1)).sum(1)
        attention_up =  (exp_attention1 * proj_feats_mask_up) # span, batch, seqlen, 1
        attention_down = (exp_attention1 * proj_feats_mask_down)

        seg_feats_att = (proj_feats * attention_up).sum(2).sum(1) / attention_down.sum(2).sum(1)

        seg_feats = torch.cat([seg_feats_edge, seg_feats_att], -1)
        seg_feats = self.att_pool_output(seg_feats)
        return seg_feats


class SegmentEncoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, add_search_word, separate, source, setting, bert_size):
        super(SegmentEncoder, self).__init__()
        self.setting = setting
        self.source = source
        self.qid2knowledge = pickle.load(open('qid2knowledge_cleaned_mavex.pkl', 'rb'))
        self.qid2seg_spans = {}
        self.small_bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        self.num_search_per_segment = 3
        self.num_head = 4
        self.max_segment = 4
        self.feat_dim = 512
        self.separate = separate
        self.knowledge_bert = SequenceEncoder(add_search_word, bert_size)
        self.label2ans = pickle.load(open('data/okvqa_new/cache/trainval_label2ans_729.pkl', 'rb'))
        self.attentive_pooling = AttentivePooling(768, hidden_dim)
        self.sentence_feat_dim = 512
        self.qid2q = pickle.load(open('qid2q.pkl', 'rb'))
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_sentences_per_sw = 4

        self.sga_q = SGASimple(self.num_head, hidden_dim, output_dim, dropout=0.2, input_dim1=hidden_dim, input_dim2=hidden_dim)
        self.sga_default = SGASimple(self.num_head, hidden_dim, output_dim, dropout=0.2, input_dim1=768, input_dim2=hidden_dim)
        self.sga_mavex = SGASimple(self.num_head, hidden_dim, output_dim, dropout=0.2, input_dim1=1024, input_dim2=hidden_dim)
        if not os.path.isfile('segment2span.pkl'):
            self.qid2seg_spans = {}
            qids = self.qid2q.keys()
            for qid in qids:
                _ = self.segment2span(qid)
            pickle.dump(self.qid2seg_spans, open('segment2span.pkl', 'wb'))
        else:
            self.qid2seg_spans = pickle.load(open('segment2span.pkl', 'rb'))

    def segment2span(self, question_id, max_question_len=22):
        if question_id in self.qid2seg_spans:
            return self.qid2seg_spans[question_id]
        segment_list = self.qid2knowledge[int(question_id)]['segment_list']
        question = self.qid2q[int(question_id)]
        question_tokens = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(question))[:max_question_len]
        segments_spans = {}
        for segment in segment_list:
            segment_tokens = self.small_bert_tokenizer.convert_tokens_to_ids(
                self.small_bert_tokenizer.tokenize(segment.replace('||default||', '').replace('MAVEx', '')))
            found = -1
            for i in range(len(question_tokens) - len(segment_tokens) + 1):
                if question_tokens[i:i + len(segment_tokens)] == segment_tokens:
                    found = i
            if found >= 0:
                segments_spans[segment] = [found, found + len(segment_tokens)]
            else:
                segments_spans[segment] = [-1, -1]
        self.qid2seg_spans[question_id] = segments_spans
        return segments_spans

    def forward(self, knowledge_token, textual_features, vilbert_features, question_ids, verif_inds):

        seq_feats = self.knowledge_bert(knowledge_token)
        segment_search_word2sentences_infos, sentences_feats = self.process_segment_search_words(seq_feats, question_ids)

        batch_size = len(question_ids)
        batch_idx = []
        output_segment_spans = []

        dummy_idx = len(sentences_feats)
        sentences_feats = torch.cat([sentences_feats, torch.zeros((1, self.feat_dim)).float().cuda()])
        output_value_gather_idx = []
        output_default_value_gather_idx = []
        output_q_gather_idx = []

        for i in range(batch_size):
            segment_spans = self.segment2span(int(question_ids[i]))
            segment_list = list(segment_spans.keys())
            segment2search_words = self.qid2knowledge[int(question_ids[i])]['segments2search_words']
            assert '||default||' in segment_list[0]
            q_gather_idx = [9999] * self.max_segment
            q_seg_ct = 0
            for segment in segment_list:
                if '||default||' in segment:
                    search_words = segment2search_words[segment]
                    ct = 0
                    gather_idx = [dummy_idx] * self.num_search_per_segment
                    for sw in search_words:
                        if ct == self.num_search_per_segment:
                            break
                        if (segment + '//' + sw) in segment_search_word2sentences_infos[i]:
                            gather_idx[ct] =  segment_search_word2sentences_infos[i][segment + '//' + sw]
                            ct += 1
                    output_default_value_gather_idx.extend(gather_idx)
                elif 'MAVEx' not in segment:
                    output_segment_spans.append(segment_spans[segment])
                    search_words = segment2search_words[segment]
                    gather_idx = [dummy_idx] * self.num_search_per_segment
                    ct = 0
                    for sw in search_words:
                        if ct == self.num_search_per_segment:
                            break
                        if (segment + '//' + sw) in segment_search_word2sentences_infos[i]:
                            gather_idx[ct] = segment_search_word2sentences_infos[i][segment + '//' + sw]
                            ct += 1
                    output_value_gather_idx.extend(gather_idx)
                    # output_values.append(value_feature)
                    if q_seg_ct < self.max_segment:
                        q_gather_idx[q_seg_ct] = len(batch_idx)
                    batch_idx.append(i)
                    q_seg_ct += 1

                else:
                    continue
            output_q_gather_idx.extend(q_gather_idx)

        output_default_key = textual_features[:, 0].unsqueeze(1)
        output_default_values = sentences_feats[output_default_value_gather_idx].view(batch_size, self.num_search_per_segment, -1)
        output_default = self.sga_default(output_default_key, output_default_values).view(batch_size, -1)

        output_key = self.attentive_pooling(textual_features, output_segment_spans, batch_idx).unsqueeze(1)
        output_values = sentences_feats[output_value_gather_idx].view(len(output_segment_spans), self.num_search_per_segment, -1)

        output_q_ = self.sga_q(output_key, output_values).squeeze(1)

        output_q_ = torch.cat([output_q_, torch.zeros((1, output_q_.size(-1))).float().cuda()])
        output_q_gather_idx_real = []
        for _ in output_q_gather_idx:
            if _ == 9999:
                output_q_gather_idx_real.append(len(output_values))
            else:
                output_q_gather_idx_real.append(_)


        output_q = output_q_[output_q_gather_idx_real].view(batch_size, self.max_segment, -1)


        output_value_gather_idx = []
        output_key_gather_idx = []

        for i in range(batch_size):
            segment2search_words = self.qid2knowledge[int(question_ids[i])]['segments2search_words']
            for j in range(verif_inds.size(1)):
                pred_answer = self.label2ans[int(verif_inds[i, j])]
                segment = pred_answer + 'MAVEx'
                search_words = segment2search_words[segment]
                ct = 0
                value_gather_idx = [dummy_idx] * self.num_search_per_segment
                for sw in search_words:
                    if ct == self.num_search_per_segment:
                        break
                    if (segment + '//' + sw) in segment_search_word2sentences_infos[i]:
                        value_gather_idx[ct] = segment_search_word2sentences_infos[i][segment + '//' + sw]
                        ct += 1
                output_value_gather_idx.extend(value_gather_idx)
                output_key_gather_idx.append(i)

        output_mavex_key = vilbert_features[output_key_gather_idx].view(batch_size * 5, 1, -1) # self.mavex_bert(output_mavex_key)
        output_mavex_values = sentences_feats[output_value_gather_idx].view(batch_size * 5, self.num_search_per_segment, -1)
        output_mavex = self.sga_mavex(output_mavex_key, output_mavex_values).view(batch_size, 5, -1)
        output_mavex_mask = output_mavex_values.abs().sum(-1).sum(-1).view(batch_size, 5)
        output_mavex_mask = (output_mavex_mask > 0.).float()

        return output_q, output_default, output_mavex, output_mavex_mask

    def process_segment_search_words(self, feats, question_ids):
        # feats: (batch, num_sentences, 768)
        batch_size, num_sentences, feat_dim = feats.size()
        feats = feats.view(-1, feats.size(-1))
        segment_search_word2sentences_infos = []
        ct = 0
        sw_gather_idxs = []
        sent_id = 0
        for question_id in question_ids:
            segment_search_word2sentences_info = {}
            if self.source == 'wiki':
                tmp = self.qid2knowledge[int(question_id)]['segment_search_word2non_separate_wiki_sentences_id']
            else:
                tmp = self.qid2knowledge[int(question_id)]['segment_search_word2non_separate_concept_strs_id']
            segment_search_words = list(tmp.keys())
            for segment_search_word in segment_search_words:
                sentence_feat_ids = tmp[segment_search_word]
                gather_idxs = [batch_size * num_sentences] * self.max_sentences_per_sw
                if len(sentence_feat_ids):
                    ctt = 0
                    for sentence_feat_id in sentence_feat_ids:
                        if sentence_feat_id < num_sentences:
                            gather_idxs[ctt] = ct * num_sentences + sentence_feat_id
                            ctt += 1
                        if ctt == self.max_sentences_per_sw:
                            break
                sw_gather_idxs.extend(gather_idxs)
                segment_search_word2sentences_info[segment_search_word] = sent_id
                sent_id += 1
            ct += 1
            segment_search_word2sentences_infos.append(segment_search_word2sentences_info)
        
        tmp_sentences = torch.cat([feats, torch.zeros((1, feats.size(-1))).float().cuda()])
        sentences_feats = tmp_sentences[sw_gather_idxs].view(sent_id, self.max_sentences_per_sw, -1)
        masks = (sentences_feats.sum(-1) == 0).view(sent_id, self.max_sentences_per_sw, 1) + 0.000000001
        sentences_feats = sentences_feats.sum(1) / masks.sum(1)

        return segment_search_word2sentences_infos, sentences_feats


class SegmentAnswerPredictor(nn.Module):
    def __init__(self, num_hid):
        super(SegmentAnswerPredictor, self).__init__()
        self.sga = SGASimple(4, num_hid, num_hid, 0.15, num_hid, num_hid)
        self.residual_proj = nn.Linear(num_hid, num_hid)
        self.default_pred = SimpleClassifier(num_hid, num_hid, 5020, 0.5)
        self.answer_proj = FCNet([512, num_hid], 0.2)
        self.sigmoid = nn.Sigmoid()
        self.mavex_pred = SimpleClassifier(num_hid, num_hid, 1, 0.5)


    def forward(self, default_features, segment_features, mavex_features, ans_feats, vilbert_features):
        batch_size = default_features.size(0)
        num_answers = mavex_features.size(1)
        query_feats = default_features.view(batch_size, 1, -1)
        merged_feats = self.sga(query_feats, segment_features).view(batch_size, -1)  # batch
        merged_feats = merged_feats + self.residual_proj(vilbert_features)
        default_pred = self.default_pred(merged_feats)

        ans_feats = self.answer_proj(ans_feats).view(batch_size, num_answers, 1, -1).repeat(1, 1, num_answers, 1)
        mavex_features = mavex_features.view(batch_size, 1, num_answers, -1).repeat(1, num_answers, 1, 1)
        merged_feats_q = merged_feats.view(batch_size, 1, 1, -1)
        merged_feats_a = ans_feats * mavex_features  # batch, num_answers, num_answers
        mavex_feats = merged_feats_q * merged_feats_a
        mavex_logits = self.mavex_pred(mavex_feats).squeeze(-1)
        mavex_pred = self.sigmoid(mavex_logits)

        return default_pred, mavex_pred


class WordEmbedding(nn.Module):
    """Word Embedding
    with the definition in Dictionary.
    """

    def __init__(self, np_file='answer_embs_glove.pkl', output_dim = 512, dropout=0.3):
        super(WordEmbedding, self).__init__()
        import pickle
        self.np_file = np_file
        weight_init = torch.from_numpy(pickle.load(open(np_file, 'rb'))).float()
        self.ntoken = weight_init.size(0)
        self.emb_dim = weight_init.size(1)
        self.emb = nn.Embedding(self.ntoken, 300, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.proj = FCNet([300, output_dim],dropout=0.15)

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return self.proj(emb)

    def init_weight_dict(self):
        weight_init = torch.from_numpy(pickle.load(open(self.np_file, 'rb'))).float()
        self.ntoken = weight_init.size(0)
        self.emb_dim = weight_init.size(1)
        for i in range(self.ntoken):
            if weight_init[i].sum() == 0:
                continue
            self.emb.weight.data[i] = weight_init[i]


class VILBertForVLTasks(BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_prob=0.1):
        super(VILBertForVLTasks, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.vil_prediction_backbone = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, num_labels, 0.5)
        print('Number of Output LABELS:', self.num_labels)
        
        self.num_sentences = config.num_wiki_sentences
        self.num_concepts = config.num_concepts

        self.num_head = config.num_head
        self.use_wiki = config.use_wiki
        self.use_concept = config.use_concept
        self.use_image = config.use_image
        self.segment_dim = config.segment_dim
        self.num_hid = config.num_hid
        self.qid2knowledge = pickle.load(open('qid2knowledge_cleaned_mavex.pkl', 'rb'))
        self.qid2statements = pickle.load(open('qid2statements.pkl', 'rb'))
        self.label2ans = pickle.load(open('data/okvqa_new/cache/trainval_label2ans_729.pkl', 'rb'))
        self.add_answer_emb = config.add_answer_emb

        if self.use_image or self.use_concept or self.use_wiki:
            self.answer_statement_encoder = SequenceEncoder(False, config.bert_size, config.num_hid)
            if self.add_answer_emb:
                self.answer_embs = WordEmbedding()
        if self.use_wiki:
            self.wiki_knowledge_encoder = SegmentEncoder(self.num_hid, self.segment_dim, config.use_search_word, config.separate, 'wiki', setting = config.setting, bert_size=config.bert_size)
            self.wiki_ans_pred = SegmentAnswerPredictor(1024)
        if self.use_concept:
            self.concept_knowledge_encoder = SegmentEncoder(self.num_hid, self.segment_dim, config.use_search_word,  config.separate, 'concept', setting = config.setting, bert_size=config.bert_size)
            self.concept_ans_pred = SegmentAnswerPredictor(1024)
        if self.use_image:
            self.image_knowledge_encoder = ImageSegmentEncoder(self.num_hid, self.segment_dim)
            self.image_ans_pred = SegmentAnswerPredictor(1024)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)
        self.small_bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")

    def extract_statements(self, question_ids, verif_inds, statement_max_len = 14):
        tokens = []
        for i in range(len(question_ids)):
            statement_dict = self.qid2statements[int(question_ids[i])]
            for j in range(5):
                pred_answer = self.label2ans[int(verif_inds[i, j])]
                statement = statement_dict[pred_answer]
                statement_tokens = [101] + self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(pred_answer)) + [102]
                statement_tokens1 = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(statement))
                statement_tokens1 = statement_tokens1[:statement_max_len - len(statement_tokens) - 1] + [102]
                statement_tokens = statement_tokens + statement_tokens1
                if len(statement_tokens) < statement_max_len:
                    padding = [0] * (statement_max_len - len(statement_tokens))
                    statement_tokens = statement_tokens + padding
                tokens.append(statement_tokens)
        tokens = torch.from_numpy(np.array(tokens).astype('int')).cuda().long()
        return tokens

    def forward(
            self,
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            wiki_tokens=None,  # batch, 4(search words), 10 (sents) 768 (feat)
            concept_tokens=None,  # batch, 4(search words), 10 (sents) 768 (feat)
            google_images=None,  # batch, 5(ans), 5 (images) 2048 (feat)
            question_ids=None,
            image_attention_mask=None,
            co_attention_mask=None,
            task_ids=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
            verif_inds=None,
            ):

        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            co_attention_mask,
            task_ids,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        batch_size = input_txt.size(0)
        pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        vil_prediction = self.vil_prediction_backbone(pooled_output)

        if self.use_image or self.use_concept or self.use_wiki:
            answer_feats = self.answer_statement_encoder(self.extract_statements(question_ids, verif_inds)).view(batch_size,
                                                                                                             5, -1)
            if self.add_answer_emb:
                answer_emb = self.answer_embs(verif_inds.long())
                answer_feats += answer_emb
        if self.use_wiki:
            wiki_segments, wiki_default, wiki_mavex, wiki_mavex_mask = self.wiki_knowledge_encoder(wiki_tokens,
                                                                            sequence_output_t,
                                                                            pooled_output,
                                                                            question_ids,
                                                                            verif_inds)  # wiki_feats: batch search sentences 768

            wiki_pred, wiki_mavex_pred = self.wiki_ans_pred(wiki_default, wiki_segments, wiki_mavex,  answer_feats, pooled_output)

        else:
            wiki_pred = torch.zeros((batch_size, self.num_labels)).float().cuda() - 10000
            wiki_mavex_pred =  torch.zeros((batch_size, 5, 5)).float().cuda()
            wiki_mavex_mask = torch.zeros((batch_size, 5)).float().cuda()

        if self.use_concept:
            concept_segments, concept_default, concept_mavex, concept_mavex_mask = self.concept_knowledge_encoder(concept_tokens,
                                                                                            sequence_output_t,
                                                                                            pooled_output,
                                                                                            question_ids,
                                                                                            verif_inds)  # wiki_feats: batch search sentences 768

            concept_pred, concept_mavex_pred = self.concept_ans_pred(concept_default, concept_segments, concept_mavex,  answer_feats, pooled_output)
        else:
            concept_pred = torch.zeros((batch_size, self.num_labels)).float().cuda() - 10000
            concept_mavex_pred =  torch.zeros((batch_size, 5, 5)).float().cuda()
            concept_mavex_mask = torch.zeros((batch_size, 5)).float().cuda()

        if self.use_image:
            image_segments, image_default, image_mavex, image_mavex_mask = self.image_knowledge_encoder(sequence_output_t, sequence_output_v,
                                                                                  pooled_output, google_images,
                                                                                  question_ids)
            image_pred, image_mavex_pred = self.image_ans_pred(image_default, image_segments, image_mavex, answer_feats, pooled_output)

        else:
            image_pred = torch.zeros((batch_size, self.num_labels)).float().cuda() - 10000
            image_mavex_pred = torch.zeros((batch_size, 5, 5)).float().cuda()
            image_mavex_mask = torch.zeros((batch_size, 5)).float().cuda()

        ###############################################################################
        knowledge_prediction = torch.stack([wiki_pred, concept_pred, image_pred], 1).max(1)[0]  # batch, num_search, 5020
        knowledge_mavex_pred = torch.stack([wiki_mavex_pred, concept_mavex_pred, image_mavex_pred], 1).max(1)[
            0]  # batch, num_search, 5020
        knowledge_mavex_mask = torch.stack([wiki_mavex_mask, concept_mavex_mask, image_mavex_mask], 1).max(1)[
            0]  # batch, num_search, 5020

        return vil_prediction, knowledge_prediction, wiki_pred, concept_pred, image_pred, knowledge_mavex_pred,\
               wiki_mavex_pred, concept_mavex_pred, image_mavex_pred, knowledge_mavex_mask, wiki_mavex_mask, concept_mavex_mask, image_mavex_mask


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states) 