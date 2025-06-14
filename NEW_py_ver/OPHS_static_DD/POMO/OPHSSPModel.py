
import torch
import torch.nn as nn
import torch.nn.functional as F


class OPHSSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = OPHSSP_Encoder(**model_params)
        self.decoder = OPHSSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+hotel, EMBEDDING_DIM)
        self.stochastic_prize = model_params['stochastic_prize']

    def pre_forward(self, reset_state):
            depot_xy = reset_state.depot_xy
            # shape: (batch, hotel, 2)
            day_number = reset_state.day_number.unsqueeze(1).expand_as(depot_xy)
            # shape: (batch, hotel, 1)
            # shape: (batch, 1)
            # print(reset_state.day_number , day_number[:,:,:1])
            depot_xy_day = torch.cat((depot_xy, day_number[:,:,:1]), dim=2)
            # # shape: (batch, 1, 3)
            node_xy = reset_state.node_xy
            # shape: (batch, problem, 2)
            node_prize = reset_state.node_prize
            # shape: (batch, problem)
            if self.stochastic_prize:
                node_xy_prize = torch.cat((node_xy, node_prize), dim=2)
            else:
                node_xy_prize = torch.cat((node_xy, node_prize[:, :, None]), dim=2)
            # shape: (batch, problem, 3)

            self.encoded_nodes = self.encoder(depot_xy_day, node_xy_prize)
            # shape: (batch, problem+2, embedding)
            self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        hotel_size = state.HOTEL_IDX.size(2)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=hotel_size, end=pomo_size+hotel_size)[None, :].repeat(batch_size, 1)        # new change for batch bug fix
            selected[state.finished] = 0                                                                    
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.remaining_len, ninf_mask=state.ninf_mask)          
            # shape: (batch, pomo, problem+2)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        # print(f'probs : {probs}')
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)                  #@todo fix some shape irreqularity for problem size
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class OPHSSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.stochastic_prize = self.model_params['stochastic_prize']

        self.embedding_depot = nn.Linear(3, embedding_dim)
        if self.stochastic_prize:
            self.embedding_node = nn.Linear(4, embedding_dim)
        else:
            self.embedding_node = nn.Linear(3, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy_day, node_xy_prize):
        # depot_xy.shape: (batch, hotel, 3)
        # node_xy_prize.shape: (batch, problem, 4)

        embedded_depot = self.embedding_depot(depot_xy_day)
        # shape: (batch, hotel, embedding)
        embedded_node = self.embedding_node(node_xy_prize)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+hotel, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+hotel, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)
        self.out3 = None
    def forward(self, input1):
        # input1.shape: (batch, problem+2, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # Wqkv shape: (batch, problem+2, head_num*qkv_dim)
        # qkv shape: (batch, head_num, problem+2, qkv_dim)

        out_concat = flash_multi_head_attention(q, k, v)
        # shape: (batch, problem+2, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem+2, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem+2, embedding)


########################################
# DECODER
########################################

class OPHSSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+2, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+2, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+2)

    # def set_q1(self, encoded_q1):
    #     # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
    #     head_num = self.model_params['head_num']
    #     self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
    #     # shape: (batch, head_num, n, qkv_dim)

    # def set_q2(self, encoded_q2):
    #     # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
    #     head_num = self.model_params['head_num']
    #     self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
    #     # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, pomo, EMBEDDING_DIM+2)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = flash_multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, pomo, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)    

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

def flash_multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, dropout_rate=0):
    """
    Multi-head attention using Flash Attention for speed and efficiency.
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        rank2_ninf_mask: Rank-2 mask of shape (batch_size, seq_len)
        rank3_ninf_mask: Rank-3 mask of shape (batch_size, seq_len, seq_len)
        dropout_rate: Dropout rate applied to attention weights

    Returns:
        Tensor of shape (batch_size, seq_len, num_heads * head_dim)
    """
    # Ensure tensors are contiguous
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # Prepare attention mask
    mask = None
    if rank2_ninf_mask is not None:
        mask = rank2_ninf_mask[:, None, None, :].to(q.device)
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :].to(q.device) if mask is None else mask + rank3_ninf_mask[:, None, :, :].to(q.device)

    # Use flash attention for speed and efficiency
    with torch.backends.cuda.sdp_kernel(enable_flash=True):
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_rate)

    # Reshape output
    batch_size, num_heads, seq_len, head_dim = q.size()
    output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    return output



class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


# class AddAndBatchNormalization(nn.Module):
#     def __init__(self, **model_params):
#         super().__init__()
#         embedding_dim = model_params['embedding_dim']
#         self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
#         # 'Funny' Batch_Norm, as it will normalized by EMB dim

#     def forward(self, input1, input2):
#         # input.shape: (batch, problem, embedding)

#         batch_s = input1.size(0)
#         problem_s = input1.size(1)
#         embedding_dim = input1.size(2)

#         added = input1 + input2
#         normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
#         back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

#         return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))