import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm

class EstimatorCV():
    def __init__(self, feature_num, class_num, beta):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.beta = beta
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()

    # token_features: (doc_num, k+1, feature_dim)
    # fake_labels: (doc_num, class_num) 
    # selected_weights: (doc_num, k+1)
    def update_CV(self, token_features, fake_labels, selected_weights):
        N = token_features.size(0)
        C = self.class_num
        tokens_num = token_features.size(1)
        dim = token_features.size(2)
       
        weights = selected_weights.view(N, 1, tokens_num)
        # sum1: (doc_num, 1, dim) -> (doc_num, dim)
        sum1 = torch.matmul(weights, token_features).view(N, dim)
        # sum2: (class_num, dim)
        sum2 = torch.matmul(fake_labels.t(), sum1)
        # norm_term: (class_num)
        norm_term =  torch.matmul(fake_labels.t(), selected_weights.sum(dim=1))
        # mu: (class_num, dim)
        mu = sum2 / norm_term.view(C, 1)

        sigma = torch.zeros(C, dim, dim).cuda()
        dataset = TensorDataset(token_features, fake_labels, selected_weights)
        dataset_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        for step, batch in enumerate(tqdm(dataset_loader)):
            batch_token_features = batch[0]
            batch_fake_labels = batch[1]
            batch_selected_weights = batch[2]

            set_batch_size = batch_token_features.size(0)
            batch_temp1 = batch_token_features.view(set_batch_size, tokens_num, 1, dim).expand(set_batch_size, tokens_num, C, dim) - mu.view(1,1,C,dim).expand(set_batch_size, tokens_num, C, dim)
            # transposed_temp1 = temp1.permute(0,1,3,2)
            # temp2: (doc_num, tokens_num, dim, dim)
            batch_temp2 = torch.matmul(batch_temp1.permute(0,1,3,2), batch_temp1)
            # temp3: (doc_num, tokens_num, dim, dim)
            batch_temp3 = torch.mul(batch_selected_weights.view(set_batch_size,tokens_num,1,1).expand(set_batch_size,tokens_num,dim,dim), batch_temp2)
            # temp4: (doc_num, dim, dim)
            batch_temp4 = batch_temp3.sum(dim=1)
            batch_temp5 = torch.mul(batch_fake_labels.t().view(C,set_batch_size,1,1).expand(C,set_batch_size,dim,dim), batch_temp4)
            # sigma: (class_num, dim, dim)
            batch_sigma = batch_temp5.sum(dim=1)
            sigma = sigma + batch_sigma
        
        
        # temp1 = token_features.view(N, tokens_num, 1, dim).expand(N, tokens_num, C, dim) - mu.view(1,1,C,dim).expand(N, tokens_num, C, dim)
        # # transposed_temp1 = temp1.permute(0,1,3,2)
        # # temp2: (doc_num, tokens_num, dim, dim)
        # temp2 = torch.matmul(temp1.permute(0,1,3,2), temp1)
        # # temp3: (doc_num, tokens_num, dim, dim)
        # temp3 = torch.mul(selected_weights.view(N,tokens_num,1,1).expand(N,tokens_num,dim,dim), temp2)
        # # temp4: (doc_num, dim, dim)
        # temp4 = temp3.sum(dim=1)
        # temp5 = torch.mul(fake_labels.t().view(C,N,1,1).expand(C,N,dim,dim), temp4)
        # # sigma: (class_num, dim, dim)
        # sigma = temp5.sum(dim=1) / norm_term.view(C,1,1)

        sigma = sigma / norm_term.view(C,1,1)

        self.CoVariance = (1-self.beta) * sigma + self.beta * self.CoVariance