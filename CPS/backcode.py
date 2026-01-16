class TeacherNicheAttention(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, num_heads, dropout, share_weights=False, prep_scale=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.share_weights = share_weights
        self.prep_scale = prep_scale
        
        if not prep_scale:
            self.multi_scale_convs = MultiHopSSGConv(
                in_dim, out_dim, k_list, dropout)
        else:
            self.gene_proj = nn.Linear(in_dim, out_dim)
        
        if share_weights:
            self.query_proj = nn.Linear(out_dim, out_dim)
            self.key_proj = nn.Linear(out_dim, out_dim)
            self.value_proj = nn.Linear(out_dim, out_dim)
        else:
            self.query_projs = nn.ModuleList([
                nn.Linear(out_dim, out_dim) for _ in k_list
            ])
            self.key_projs = nn.ModuleList([
                nn.Linear(out_dim, out_dim) for _ in k_list
            ])
            self.value_projs = nn.ModuleList([
                nn.Linear(out_dim, out_dim) for _ in k_list
            ])
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.residual = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(out_dim)
        self.norm2 = LayerNorm(out_dim)
        
    def forward(self, x, edge_index=None):
        N = x.shape[0]
        if not self.prep_scale:
            multi_scale_features = self.multi_scale_convs(x, edge_index) # list[(N, D)]
            scale_features = torch.stack(multi_scale_features, dim=1) # (N, S, D)
        else:
            multi_scale_features = x
            scale_features = self.gene_proj(multi_scale_features)  # (N, S, D)
        
        # Pre-LN before attention
        # normed_features = self.norm(scale_features)
        normed_features = scale_features
        
        if self.share_weights:
            query = self.query_proj(normed_features[:,0,:]).reshape(N, self.num_heads, self.head_dim) # (N, H, D_h)
            keys = self.key_proj(normed_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
            values = self.value_proj(normed_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
        else:
            queries, keys, values = [], [], []
            for i, (q_proj, k_proj, v_proj) in enumerate(zip(self.query_projs, self.key_projs, self.value_projs)):
                q = q_proj(normed_features[:,0,:]).reshape(N, self.num_heads, self.head_dim)
                k = k_proj(normed_features[:,i,:]).reshape(N, self.num_heads, self.head_dim)
                v = v_proj(normed_features[:,i,:]).reshape(N, self.num_heads, self.head_dim)
                queries.append(q)  # i+ [(N, H, D_h)]
                keys.append(k)
                values.append(v)
            query = torch.mean(torch.stack(queries, dim=1), dim=1)  # (N, H, D_h)
            keys = torch.stack(keys, dim=1)  # (N, S, H, D_h)
            values = torch.stack(values, dim=1)  # (N, S, H, D_h)
        
        attn_scores = torch.einsum('nhd,nshd->nsh', query, keys) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, S, H)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.einsum('nsh,nshd->nhd', attn_weights, values)
        output = attended.reshape(N, -1)  # (N, l_dim)
        
        # output = output + residual
        residual = self.residual(scale_features[:,0,:])
        output = output + residual
        
        # projection
        output = self.out_proj(output)
        
        # output = self.norm2(output)
        return output, attn_weights
    
    
    
# TeacherNetwork
class TeacherNicheAttention(nn.Module):
    def __init__(self, in_dim, out_dim, k_list, num_heads, dropout, share_weights=False, prep_scale=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.share_weights = share_weights
        self.prep_scale = prep_scale
        
        if not prep_scale:
            self.multi_scale_convs = MultiHopSGConv(in_channels=in_dim, 
                                                    out_channels=out_dim,
                                                    k_list=k_list,
                                                    prep_scale=prep_scale,
                                                    dropout=dropout)
        else:
            self.gene_proj = nn.Linear(in_dim, out_dim)
            self.activation = nn.GELU()
            self.proj_dropout = nn.Dropout(dropout)
        
        self.query_proj = nn.Linear(out_dim, out_dim)
        self.key_proj = nn.Linear(out_dim, out_dim)
        self.value_proj = nn.Linear(out_dim, out_dim)
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm_input = nn.LayerNorm(out_dim)
        self.norm_post = nn.LayerNorm(out_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.log_tau = nn.Parameter(torch.tensor(math.log(0.5)))
        
        self.scale_type = nn.Parameter(torch.randn(1, self.num_scales, out_dim) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.identity_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim), 
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, x, edge_index=None):
        N = x.shape[0]
        
        if not self.prep_scale:
            multi_scale_features = self.multi_scale_convs(x, edge_index) # list[(N, D)]
            scale_features = torch.stack(multi_scale_features, dim=1) # (N, S, D)
        else:
            scale_features = self.proj_dropout(self.activation(self.gene_proj(x)))  # (N, S, D)

        # self.check_token_similarity(scale_features)
        scale_features = scale_features + self.scale_type
        scale_features = self.norm_input(scale_features)
        
        self_feature = scale_features[:, 0, :]
        query = self.query_proj(self_feature).reshape(N, self.num_heads, self.head_dim)
        
        # global_context = torch.mean(scale_features, dim=1) # (N, D)
        # query = self.query_proj(global_context).reshape(N, self.num_heads, self.head_dim) # (N, H, D_h)
        keys = self.key_proj(scale_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
        values = self.value_proj(scale_features).reshape(N, self.num_scales, self.num_heads, self.head_dim)
        
        tau = torch.exp(self.log_tau)
        attn_scores = torch.einsum('nhd,nshd->nsh', query, keys) / ((self.head_dim ** 0.5) * tau)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, S, H)
        
        avg_weights = attn_weights.mean(dim=2)
        entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-9), dim=1).mean()
        er_loss = 0.01 * F.relu(entropy - 1.0)
        print(er_loss.item(), entropy.item(), f"Current Tau: {tau.item()}")
        
        attn_weights = self.dropout(attn_weights)
        context = torch.einsum('nsh,nshd->nhd', attn_weights, values).reshape(N, -1)
        context = self.out_proj(context)
        # output = self.identity_proj(context)
        self_id = self.identity_proj(self_feature)
        output = self_id + context
        # output = self.identity_proj(self_feature + context)
        
        output = self.norm_post(output)

        return output, attn_weights, er_loss