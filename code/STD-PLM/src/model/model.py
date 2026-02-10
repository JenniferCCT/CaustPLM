from typing import Iterator, Mapping
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from utils.utils import norm_Adj, lap_eig, topological_sort
from model.sandglassAttn import SAG
import numpy as np
from model.position import PositionalEncoding
from model.basic import PromptPoolPrefix


class DecodingLayer(nn.Module):

    def __init__(self, input_dim ,emb_dim, output_dim):
        super().__init__()

        hidden_size = (emb_dim+output_dim)*2//3
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        #nn.Linear(in_features=input_dim,out_features=output_dim)
        
    def forward(self, llm_hidden):

        out = self.fc(llm_hidden)

        return out

class TimeEmbedding(nn.Module):

    def __init__(self,t_dim):
        super().__init__()

        #self.hour_embedding = nn.Embedding(num_embeddings=24,embedding_dim=t_dim)
        self.day_embedding = nn.Embedding(num_embeddings=288,embedding_dim=t_dim)
        self.week_embedding = nn.Embedding(num_embeddings=7,embedding_dim=t_dim)

    def forward(self,TE):

        # TE (B,T,5)

        B,T,_ = TE.shape

        week = (TE[...,2].to(torch.long) % 7).view(B*T,-1)
        hour = (TE[...,3].to(torch.long) % 24).view(B*T,-1)
        minute = (TE[...,4].to(torch.long) % 60).view(B*T,-1)

        DE = self.day_embedding((hour*60+minute)//5)
        #HE = self.hour_embedding(hour)
        WE = self.week_embedding(week)

        te = torch.concat((DE,WE),dim=-1).view(B,T,-1)

        return te


class NodeEmbedding(nn.Module):
    def __init__(self, adj_mx, node_emb_dim, k = 16, dropout = 0 ):
        super().__init__()
        N,_ = adj_mx.shape
        self.k = k

        self.setadj(adj_mx=adj_mx)

        self.fc = nn.Linear(in_features=k,out_features=node_emb_dim)

    def forward(self):

        node_emgedding = self.fc(self.lap_eigvec)

        return node_emgedding
    
    def setadj(self,adj_mx):
        N,_ = adj_mx.shape

        self.adj_mx = adj_mx

        eigvec, eigval = lap_eig(self.adj_mx)
        k = self.k
        if k>N:
            eigvec = np.concatenate((eigvec, np.zeros(N,k-N)), dim = -1)
            eigval = np.concatenate((eigval, np.zeros(k-N)), dim = -1)
        
        ind = np.abs(eigval).argsort(axis=0)[::-1][:k]

        eigvec = eigvec[:, ind]        

        if hasattr(self,'lap_eigvec'):
            self.lap_eigvec = torch.tensor(eigvec).float()
        else :
            self.register_buffer('lap_eigvec', torch.tensor(eigvec).float())
    
class Time2Token(nn.Module):
    def __init__(self,sample_len, features, emb_dim, tim_dim, dropout):
        super().__init__()
        
        self.sample_len = sample_len

        in_features =  sample_len*features*2 + tim_dim
        hidden_size = (in_features + emb_dim)*2//3
        self.fc_state = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        input_dim = tim_dim + (sample_len-1)*features*2
        hidden_size = (input_dim+emb_dim)*2//3
        self.fc_grad = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )        

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self,x,te,mask):
        # te(B,T,tim_dim)

        B,N,TF = x.shape

        x = x.view(B,N,self.sample_len,-1) #B,N,T,F
        x = torch.concat((x,mask.view(B,N,self.sample_len,-1)),dim=-1)
        x = x.mean(dim=1) #B,T,F

        state = x.view(B,1,-1)
        state = torch.concat((state,te[:,-1:,:]),dim=-1)#(B,1,TF+tim_dim)
        state = self.fc_state(state)

        grad = (x[:,1:,:] - x[:,:-1,:]).view(B,1,-1)#(B,1,(T-1)F)
        grad = torch.concat((grad,te[:,-1:,:]),dim=-1)#(B,1,(T-1)F+tim_dim)
        grad = self.fc_grad(grad)

        out = torch.concat((state,grad),dim=1)

        out = self.ln(out)

        return out


class Node2Token(nn.Module):
    def __init__(self,sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()

        in_features = sample_len*features*2
        
        self.use_node_embedding = use_node_embedding

        state_features =  tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, emb_dim),
        )
        
        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,emb_dim),
        )

        self.mask_token = nn.Linear(in_features=sample_len*features,out_features=emb_dim)

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self,x,te,ne,mask):
        #te(B,T,tim_dim)  ne(N,node_emb_dim) mask(B,T,N,F)
        B,N,TF = x.shape

        mask = mask.permute(0,2,1,3).contiguous().view(B,N,-1) #(B,N,TF)
        x = torch.concat((x,mask),dim=-1)

        state = te[:,-1:,:].repeat(1,N,1)
        if self.use_node_embedding:
            ne = torch.unsqueeze(ne,dim=0).repeat(B,1,1)
            state = torch.concat((state,ne),dim=-1)
        state = self.state_fc(state)

        x = self.fc1(x) #B,N,D
        x += self.mask_token(mask) #test

        out = state + x

        out = self.ln(out)

        return out

class SCALiteQueryGate(nn.Module):
    """
    Query-aware SCA-lite gate.
    beta_i = sigmoid(MLP([token_i || query]))
    token_i: (B, N, D)
    query:   (B, D)
    beta:    (B, N, 1)
    """
    def __init__(self, emb_dim: int, hidden_mult: float = 0.5):
        super().__init__()
        hidden = max(32, int(emb_dim * hidden_mult))
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, N, D = node_tokens.shape
        q = query.unsqueeze(1).expand(B, N, D)          # (B, N, D)
        x = torch.cat([node_tokens, q], dim=-1)         # (B, N, 2D)
        beta = torch.sigmoid(self.mlp(x))               # (B, N, 1)
        return beta



class STALLM(nn.Module):
    def __init__(self,basemodel,sample_len, output_len,\
                 input_dim , output_dim , 
                  node_emb_dim , sag_dim, sag_tokens, \
                 adj_mx = None, dis_mx = None , use_node_embedding = True,\
                 use_timetoken = True, use_sandglassAttn = True, \
                 dropout = 0, trunc_k = 16, t_dim = 64,wo_conloss=False,\
                 prompt_pool: bool = False,pool_size: int = 30,top_k: int = 3, \
                 prompt_len: int = 5,prompt_sim_lambda: float = 1e-3,cross_attn:bool = False):
        super().__init__()

        self.topological_sort_node = True

        tim_dim = t_dim *2 #hour,week    

        self.setadj(adj_mx,dis_mx)

        self.output_dim = output_dim
        self.input_dim = input_dim

        self.emb_dim = basemodel.emb_dim
        self.basemodel = basemodel

        self.sample_len = sample_len
        self.output_len = output_len
        self.sag_tokens = sag_tokens

        self.use_prompt_pool = prompt_pool
        self.prompt_sim_lambda = prompt_sim_lambda
        if self.use_prompt_pool:
            self.prompt_pool = PromptPoolPrefix(
                emb_dim=self.emb_dim,
                pool_size=pool_size,
                top_k=top_k,
                prompt_len=prompt_len
            )
        self.use_cross_attn = cross_attn
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm3 = nn.LayerNorm(self.emb_dim)  # for S->T
        self.layer_norm4 = nn.LayerNorm(self.emb_dim)  # for T->S
        self.layer_norm5 = nn.LayerNorm(self.emb_dim)

        self.prompt_proj = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim))

        self.use_sca_lite = True
        self.sca_eps = 0.2
        self.gate_lambda = 1e-3   # 可選：讓 gate 稍微稀疏
        self.sca_lite_gate = SCALiteQueryGate(self.emb_dim, hidden_mult=0.5)

        self.use_sandglassAttn = use_sandglassAttn
        if use_sandglassAttn:
            self.wo_conloss = wo_conloss
            self.sandglassAttn = SAG(sag_dim=sag_dim, sag_tokens=sag_tokens, emb_dim=self.emb_dim, sample_len=sample_len, features=input_dim ,dropout=dropout)


        self.spatialTokenizer =  Node2Token(sample_len=sample_len,features=input_dim,node_emb_dim=node_emb_dim,\
                                            emb_dim=self.emb_dim, \
                                            tim_dim=tim_dim,dropout=dropout,use_node_embedding=use_node_embedding)

        self.out_mlp = DecodingLayer(input_dim=output_dim*sample_len, \
                                     emb_dim=self.emb_dim, \
                                     output_dim=output_dim*output_len)

        self.timeembedding = TimeEmbedding(t_dim=t_dim)

        self.use_node_embedding = use_node_embedding
        if use_node_embedding:
            self.node_embd_layer = NodeEmbedding(adj_mx=adj_mx,node_emb_dim=node_emb_dim, k=trunc_k, dropout=dropout)

        self.use_timetoken = use_timetoken
        if use_timetoken:
            self.timeTokenizer = Time2Token(sample_len=sample_len,features=input_dim,\
                                            emb_dim=self.emb_dim, \
                                            tim_dim=tim_dim,dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(self.emb_dim)


    # def forward(self,x:torch.FloatTensor,timestamp:torch.Tensor,prompt_prefix:Optional[torch.LongTensor],mask:torch.LongTensor):
    #     other_loss = []

    #     # timestamp (B,T,4)
    #     timestamp = timestamp[:,:self.sample_len,:]

    #     B,N,TF = x.shape #(Batch,N,T*features)
    #     # emb of time
    #     te = self.timeembedding(timestamp) #(B,T,tim_dim)
    #     # emb of nodes
    #     if self.use_node_embedding:
    #         ne = self.node_embd_layer()
    #     else:
    #         ne = None

    #     # spatial token
    #     spatial_token = self.spatialTokenizer(x,te,ne,mask)
    #     if self.topological_sort_node:
    #         spatial_token = spatial_token[:,self.node_order,:]

    #     # spatial -> sandglassAttn
    #     st_embeddin g = spatial_token
    #     s_num = N
    #     if self.use_sandglassAttn:
    #         s_num = self.sag_tokens
    #         st_embedding,attn_weights = self.sandglassAttn.encode(st_embedding) #(B,N',D) #attn_weights(B,N',N)
    #         if not self.wo_conloss:
    #             scale = attn_weights.sum(dim=1)#(B,N)

    #             sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
    #             other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)

    #             Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
    #             other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())

    #     if self.use_timetoken:
    #         time_tokens = self.timeTokenizer(x,te,mask)
    #         time_tokens_idx = st_embedding.shape[1]
    #         st_embedding = torch.concat([time_tokens,st_embedding],dim=1)


    #     if self.use_prompt_pool:
    #         query = st_embedding.mean(dim=1)   # (B, D)
    #         prompt_tokens, sim_loss, _ = self.prompt_pool(query)
    #         st_embedding = torch.cat([prompt_tokens, st_embedding], dim=1)
    #         other_loss.append(self.prompt_sim_lambda * sim_loss)
    #     else:
    #         # keep old behavior if you still want ablation with prompt_prefix
    #         if prompt_prefix is not None:
    #             prompt_len,_ = prompt_prefix.shape
    #             prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
    #             prompt_embedding = prompt_embedding.repeat(B,1,1)
    #             st_embedding = torch.concat([prompt_embedding,st_embedding],dim=1)


    #     hidden_state = st_embedding

    #     hidden_state = self.basemodel(hidden_state)
    #     s_state = hidden_state[:,-s_num:,:]


    #     if self.use_sandglassAttn:
    #         s_state = self.sandglassAttn.decode(s_state,spatial_token) 
    #     s_state += spatial_token

    #     if self.topological_sort_node:
    #         s_state = s_state[:,self.node_order_rev,:]

    #     if self.use_timetoken:
    #         t_state = hidden_state[:,-time_tokens_idx-1:-time_tokens_idx,:]
    #         t_state += time_tokens[:,-1:,:]

    #         s_state += t_state

    #     s_state = self.layer_norm(s_state)

    #     out = self.out_mlp(s_state)

    #     return out, other_loss

    # =========================
    # STALLM.forward (refactor)
    # =========================
    def forward(
        self,x: torch.FloatTensor,timestamp: torch.Tensor,prompt_prefix: Optional[torch.LongTensor],mask: torch.LongTensor):
        other_loss = []

        # -------------------------
        # 0) prepare time features
        # -------------------------
        timestamp = timestamp[:, : self.sample_len, :]          # (B, T, 4)
        B, N, TF = x.shape                                      # (B, N, T*F)

        te = self.timeembedding(timestamp)                      # (B, T, tim_dim)
        ne = self.node_embd_layer() if self.use_node_embedding else None

        # -------------------------
        # 1) build spatial tokens
        # -------------------------
        spatial_token = self.spatialTokenizer(x, te, ne, mask)   # (B, N, D)
        if self.topological_sort_node:
            spatial_token = spatial_token[:, self.node_order, :] # (B, N, D)

        spatial_tokens = spatial_token                           # (B, Ts, D)  Ts=N initially

        if self.use_sca_lite:
            # query 可以用「目前所有 node token 的平均」：全局狀態摘要
            q = spatial_tokens.mean(dim=1)  # (B, D)

            beta = self.sca_lite_gate(spatial_tokens, q)         # (B, N, 1)
            spatial_tokens = self.apply_residual_gate(spatial_tokens, beta, eps=self.sca_eps)

            # (Optional) sparsity regularization: encourage smaller beta (sparser influence)
            if getattr(self, "gate_lambda", 0.0) > 0:
                other_loss.append(self.gate_lambda * beta.mean())

        s_num = N

        # optional SAG compression
        if self.use_sandglassAttn:
            s_num = self.sag_tokens
            spatial_tokens, attn_weights = self.sandglassAttn.encode(spatial_tokens)  # (B, Ts', D)
            if not self.wo_conloss:
                scale = attn_weights.sum(dim=1)  # (B, N)

                sag_score = torch.einsum('bmn,bhn->bhm', self.adj_mx[None, :, :], attn_weights)
                other_loss.append(-((sag_score * attn_weights - attn_weights * attn_weights)).sum(dim=2).mean() * 10)

                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale, dim=-1)).sum())

        # -------------------------
        # 2) build temporal tokens
        # -------------------------
        if self.use_timetoken:
            time_tokens = self.timeTokenizer(x, te, mask)        # (B, Tt, D)
        else:
            time_tokens = None

        # -------------------------
        # 3) Bi-directional Cross-Attention BEFORE LLM
        # -------------------------
        # Requires you to define in __init__:
        # self.use_cross_attn = True/False
        # self.cross_attn = nn.MultiheadAttention(self.emb_dim, num_heads=..., dropout=..., batch_first=True)
        # self.layer_norm3 = nn.LayerNorm(self.emb_dim)  # for S->T branch
        # self.layer_norm4 = nn.LayerNorm(self.emb_dim)  # for T->S branch
        # self.layer_norm  = nn.LayerNorm(self.emb_dim)  # fuse (optional; keep if you want)
        #
        # NOTE: This version keeps "two outputs" (updated time_tokens + updated spatial_tokens),
        # then concat them. This is the cleanest for your STALLM slicing logic later.

        if self.use_cross_attn and (time_tokens is not None):
            x_time = time_tokens
            x_space = spatial_tokens

            # S -> T : update temporal using spatial
            x_s_to_t, _ = self.cross_attn(query=x_time, key=x_space, value=x_space, need_weights=False)
            x_s_to_t = self.layer_norm3(x_s_to_t + x_time) 

            # T -> S : update spatial using temporal
            x_t_to_s, _ = self.cross_attn(query=x_space, key=x_time, value=x_time, need_weights=False)
            x_t_to_s = self.layer_norm4(x_t_to_s + x_space) 

            # assign back
            time_tokens = x_t_to_s
            spatial_tokens = x_s_to_t

            # (Optional) if你真的想照你貼的簡潔版「再融合一次」
            # 但注意：time_tokens 與 spatial_tokens 長度通常不同，不能直接相加。
            # 所以這個 output 融合那行在 STALLM 這種不同 token 數量的設定下不適用。
            #
            # output = self.layer_norm(x_t_to_s + x_s_to_t)  # ❌ lengths mismatch usually

        # -------------------------
        # 4) concat tokens for LLM
        # -------------------------
        if time_tokens is not None:
            # keep for later slicing if你有用到（你原本的 time_tokens_idx 其實是 spatial_tokens 長度）
            time_tokens_idx = spatial_tokens.shape[1]  # Ts'  (spatial token count after SAG)
            st_embedding = torch.cat([time_tokens, spatial_tokens], dim=1)  # (B, Tt+Ts', D)
        else:
            time_tokens_idx = None
            st_embedding = spatial_tokens                                    # (B, Ts', D)

        st_embedding = self.layer_norm5(st_embedding)

        # -------------------------
        # 5) prefix prompt injection (prompt pool OR prompt_prefix)
        # -------------------------
        if self.use_prompt_pool:
            query = st_embedding.mean(dim=1)                                # (B, D)
            prompt_tokens, sim_loss, _ = self.prompt_pool(query)            # (B, K*Lp, D)
            st_embedding = torch.cat([prompt_tokens, st_embedding], dim=1)
            other_loss.append(self.prompt_sim_lambda * sim_loss)
        else:
            if prompt_prefix is not None:
                prompt_len, _ = prompt_prefix.shape
                prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1, prompt_len, -1)
                prompt_embedding = prompt_embedding.repeat(B, 1, 1)
                # prompt_embedding = self.prompt_proj(prompt_embedding)
                st_embedding = torch.cat([prompt_embedding, st_embedding], dim=1)

        # -------------------------
        # 6) LLM forward
        # -------------------------
        hidden_state = self.basemodel(st_embedding)

        # -------------------------
        # 7) take spatial states from the tail (unchanged logic)
        # -------------------------
        s_state = hidden_state[:, -s_num:, :]   # (B, Ts', D)

        # decode SAG back to node-level if needed
        if self.use_sandglassAttn:
            s_state = self.sandglassAttn.decode(s_state, spatial_token)     # (B, N, D)
        s_state = s_state + spatial_token                                   # residual to original node tokens

        if self.topological_sort_node:
            s_state = s_state[:, self.node_order_rev, :]

        s_state = self.layer_norm(s_state)
        out = self.out_mlp(s_state)

        return out, other_loss

    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params

    def setadj(self,adj_mx,dis_mx):

        self.adj_mx = torch.tensor(adj_mx).cuda()
        self.dis_mx = torch.tensor(dis_mx).cuda()
        self.d_mx = self.adj_mx.sum(dim=1)
        N = self.adj_mx.shape[0]
        self.alpha = torch.tensor([1.05] * N).cuda() + torch.softmax(self.d_mx,dim=0)*5 
        self.node_order,self.node_order_rev = topological_sort(adj_mx)

    @staticmethod
    def apply_residual_gate(tokens: torch.Tensor, beta: torch.Tensor, eps: float = 0.2):
        scale = eps + (1.0 - eps) * beta
        return tokens * scale


# if __name__ == "__main__":
    
    # model = Phi4ST_GNN(8)

    # data = torch.rand(1,12,8)

    # print(model(x=data))

    # optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad ,model.parameters()),lr=1e-3)

    # model.save('test.pth')

    # model.load('test.pth')


