import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization, xavier_uniform_initialization
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class NGCFpretrain(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCFpretrain, self).__init__(config, dataset)

        # 추가
        self.use_img = config['img']
        self.use_txt = config['txt']
        self.use_price = config['price']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
            

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        # 추가
        if self.use_img == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_img').astype(np.float32))
            )
        elif self.use_txt == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_txt').astype(np.float32))
            )
        elif self.use_price == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_prices').astype(np.float32))
            )


    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
            np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):

        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class LightGCNpretrain(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCNpretrain, self).__init__(config, dataset)

        # 추가
        self.use_img = config['img']
        self.use_txt = config['txt']
        self.use_price = config['price']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        # int type:the layer num of lightGCN
        self.n_layers = config["n_layers"]
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        # 추가
        if self.use_img == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_img').astype(np.float32))
            )
        elif self.use_txt == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_txt').astype(np.float32))
            )
        elif self.use_price == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_prices').astype(np.float32))
            )

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class NGCFconcat(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCFconcat, self).__init__(config, dataset)
        
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
    
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]
        
        ### 추가
        self.use_img = config['img']
        self.use_txt = config['txt']
        self.use_price = config['price']
        

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        # 사이즈 늘리기 image 64개 txt ?
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # nn.Embedding.from_pretrained(
        #    torch.from_numpy(dataset.get_preload_weight('iid').astype(np.float32)))  ################## nn.Embedding(self.n_items, self.embedding_size)
        if self.use_img==True:
            self.features_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight('iid_img').astype(np.float32)))
        elif self.use_txt==True:
            self.features_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight('iid_txt').astype(np.float32)))
        elif self.use_price==True:
            self.features_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight('iid_price').astype(np.float32)))

             
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        # final prediction model YJ
        #INPUT_SIZE = 2376 # l=4 이웃들 정보를 concat한 것

    
        # INPUT_SIZE = 2*(len(self.hidden_size_list))*config["embedding_size"]+self.features_embedding.weight.shape[1] # 576
        INPUT_SIZE = sum(self.hidden_size_list)*2 + self.features_embedding.weight.shape[1]
        
        # INPUT_SIZE = config["embedding_size"]
        # 2*config["embedding_size"] + 64 + 1800 # 1992
        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )      

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
            np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):

        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )


        return user_all_embeddings, item_all_embeddings

    def concat_item_embedding(self, item_all_embeddings, n_item):
        features_embedding=self.features_embedding.weight[n_item]
        #item_image_embedding = self.item_image_embedding.weight[n_item]
        #item_text_embedding = self.item_text_embedding.weight[n_item]
        
        final_item_embedding = torch.cat([item_all_embeddings, features_embedding], dim=1)
        return final_item_embedding

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        
        pos_embeddings = item_all_embeddings[pos_item]
        pos_embeddings = self.concat_item_embedding(pos_embeddings,pos_item)
        neg_embeddings = item_all_embeddings[neg_item]
        neg_embeddings = self.concat_item_embedding(neg_embeddings,neg_item)

        #print('@ 0 ~ @ @@@@@@@@@@@@@@@@@@@@@@@@@')
        
        
        
        #pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1) # 기존의 dot product 방법 
        pos_scores = torch.cat([u_embeddings, pos_embeddings], dim=1)
        # print shape
        #print('pos_scores shape: ', pos_scores.shape) # 2048x2376  
        #print('u_embeddings shape: ', u_embeddings.shape) # 
        #print('pos_embeddings shape: ', pos_embeddings.shape) # 
        pos_scores = self.fc(pos_scores)
        #neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1) # 기존의 dot product 방법
        neg_scores = torch.cat([u_embeddings, neg_embeddings], dim=1)   
        neg_scores = self.fc(neg_scores)
        
        #print('@ 1 ~ @ @@@@@@@@@@@@@@@@@@@@@@@@@')
        
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss
        
        #print('@ 2 ~ @ @@@@@@@@@@@@@@@@@@@@@@@@@')

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings
        
        #print('@ 3 ~ @ @@@@@@@@@@@@@@@@@@@@@@@@@')

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        # concat

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        i_embeddings = self.concat_item_embedding(i_embeddings,item)
        
        # scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        # print shape
        #print('u_embeddings: ', u_embeddings.shape) # 
        #print('i_embeddings: ', i_embeddings.shape) # 
        x = torch.cat([u_embeddings, i_embeddings], dim=1)        
        scores = self.fc(x)
        # reshape 2D to 1D
        scores = scores.view(-1)
        
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        
        # # final prediction
        # x = torch.cat([u_embeddings, self.restore_item_e.transpose(0, 1)], dim=1) 
        # scores = self.fc(x)        
        
        print('@@@@@@@@@@@@@@@@@@@@@')
        print(u_embeddings.shape)
        print('@@@@@@@@@@@@@@@@@@@@@')
        print(self.restore_item_e.shape)
        print(self.restore_item_e.transpose(0, 1).shape)
        
        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class LightGCNconcat(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCNconcat, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        # int type:the layer num of lightGCN
        self.n_layers = config["n_layers"]
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        
    
        # define features embedding
        
        ### 추가
        self.img = config['img']
        self.txt = config['txt']
        self.price = config['price']
        
        if self.img==True:
            self.features_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight('iid_img').astype(np.float32)))
        elif self.txt==True:
            self.features_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight('iid_txt').astype(np.float32)))
        elif self.price==True:
            self.features_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight('iid_price').astype(np.float32)))


        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        INPUT_SIZE = self.latent_dim*2 + self.features_embedding.weight.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )      

        

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        
        return user_all_embeddings, item_all_embeddings
    
    def concat_item_embedding(self, item_all_embeddings, n_items):
        features_embedding = self.features_embedding.weight[n_items]
        final_item_embedding = torch.cat([item_all_embeddings, features_embedding], dim=1)
        return final_item_embedding

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        pos_embeddings = self.concat_item_embedding(pos_embeddings, pos_item)
        neg_embeddings = item_all_embeddings[neg_item]
        neg_embeddings = self.concat_item_embedding(neg_embeddings, neg_item)

        # calculate BPR Loss
        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        pos_scores = torch.cat([u_embeddings, pos_embeddings], dim=1)
        pos_scores = self.fc(pos_scores)
        neg_scores = torch.cat([u_embeddings, neg_embeddings], dim=1)   
        neg_scores = self.fc(neg_scores)
    
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        i_embeddings = self.concat_item_embedding(i_embeddings, item)
        
        # scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        x = torch.cat([u_embeddings, i_embeddings], dim=1)
        scores = self.fc(x)
        scores = scores.view(-1)
        
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class NGCFpretrainMLP(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCFpretrainMLP, self).__init__(config, dataset)

        # 추가
        self.use_img = config['img']
        self.use_txt = config['txt']
        self.use_price = config['price']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # 추가
        if self.use_img == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_img').astype(np.float32))
            )
        elif self.use_txt == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_txt').astype(np.float32))
            )
        elif self.use_price == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_prices').astype(np.float32))
            )

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        INPUT_SIZE = sum(self.hidden_size_list)*2
        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )     
        
        

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
            np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):

        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
    
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        
        pos_scores = torch.cat([u_embeddings, pos_embeddings], dim=1)
        pos_scores = self.fc(pos_scores)
        neg_scores = torch.cat([u_embeddings, neg_embeddings], dim=1)
        neg_scores = self.fc(neg_scores)

        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        x = torch.cat([u_embeddings, i_embeddings], dim=1) 
        scores = self.fc(x)
        # reshape 2D to 1D
        scores = scores.view(-1)
        
        # scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        
        # final prediction
        x = torch.cat([u_embeddings, self.restore_item_e.transpose(0, 1)], dim=1) 
        scores = self.fc(x)  

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class LightGCNpretrainMLP(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCNpretrainMLP, self).__init__(config, dataset)

        # 추가
        self.use_img = config['img']
        self.use_txt = config['txt']
        self.use_price = config['price']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        # int type:the layer num of lightGCN
        self.n_layers = config["n_layers"]
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        # 추가
        if self.use_img == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_img').astype(np.float32))
            )
        elif self.use_txt == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_txt').astype(np.float32))
            )
        elif self.use_price == True:
            self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_prices').astype(np.float32))
            )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
                
        INPUT_SIZE = self.latent_dim*2
        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )     


    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.cat([u_embeddings, pos_embeddings], dim=1)
        pos_scores = self.fc(pos_scores)
        neg_scores = torch.cat([u_embeddings, neg_embeddings], dim=1)
        neg_scores = self.fc(neg_scores)


        # # calculate BPR Loss
        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        x = torch.cat([u_embeddings, i_embeddings], dim=1) 
        scores = self.fc(x)
        # reshape 2D to 1D
        scores = scores.view(-1)
        
        # scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class NGCFpretrain_txt64(GeneralRecommender):
    
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCFpretrain_txt64, self).__init__(config, dataset)

        # 추가
        #self.use_img = config['img']
        #self.use_txt = config['txt']
        #self.use_price = config['price']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = 64
        self.txt_emb_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, 64)

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        self.item_embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(dataset.get_preload_weight(
                    'iid_txt').astype(np.float32)))

        self.fc = nn.Sequential(
            nn.Linear(self.txt_emb_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size),
            # nn.Sigmoid()
        )

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
            np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # make item embeddings size to 64
        #item_embeddings = self.fc(item_embeddings)

        #ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return user_embeddings, item_embeddings

    def forward(self):

        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )

        u_emb, i = self.get_ego_embeddings()
        i_emb = self.fc(i)

        all_embeddings = torch.cat([u_emb, i_emb], dim=0)

        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # print('_________________________')
        # print('user embedding size', u_embeddings.size)
        # print('item embedding size', pos_embeddings.size)

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class NGCFpretrain_all64(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NGCFpretrain_all64, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = 64
        self.txt_embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, 64)

        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        
        # 추가

        self.img_item_embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(dataset.get_preload_weight(
                'iid_img').astype(np.float32))
        )
        self.txt_item_embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(dataset.get_preload_weight(
                'iid_txt').astype(np.float32))
        )
        self.price_item_embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(dataset.get_preload_weight(
                'iid_prices').astype(np.float32))
        )

        self.fc = nn.Sequential(
            nn.Linear(self.txt_embedding_size+64+64, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size)
        )

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
            np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        txt_i_emb = self.txt_item_embedding.weight
        img_i_emb = self.img_item_embedding.weight
        price_i_emb = self.price_item_embedding.weight

        # make item embeddings size to 64
        #item_embeddings = self.fc(item_embeddings)

        #ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return user_embeddings, txt_i_emb, img_i_emb, price_i_emb

    def forward(self):

        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )

        u_emb, txt_i_emb, img_i_emb, price_i_emb = self.get_ego_embeddings()
        i_emb = torch.cat([txt_i_emb, img_i_emb, price_i_emb], dim=1)
        
        i_emb = self.fc(i_emb)

        # ㄷㅏ ㄷㅓㅎㅐ

        all_embeddings = torch.cat([u_emb, i_emb], dim=0)

        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)