import torch

class EndToEndLoss(torch.nn.Module):

    def __init__(self, w, b, loss_method, device):
        """
           Generalized End-to-End loss defined based on https://arxiv.org/abs/1710.10467
           Args:
               - init_w (float): defines the initial value of w in Equation (5) of [1]
               - init_b (float): definies the initial value of b in Equation (5) of [1]

        :param w: initial weight parameter
        :param b: initial bias parameter
       """
        super(EndToEndLoss, self).__init__()
        self.w = w
        self.b = b
        self.loss_method=loss_method
        self.device=device


    def calc_new_centroids(self, emb_vec, centroids, speaker_idx, utterance_idx):
        '''
        Returns new centroids excluding the reference utterance
        '''
        excl = torch.cat((emb_vec[speaker_idx,:utterance_idx], emb_vec[speaker_idx,utterance_idx+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == speaker_idx:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def cos_similarity(self, emb_vec, centroid):
        #embedded vector is NxMxEmbeddedFeatures where N is speakers and M is utterances
        cos_sim_matrix = []
        for speaker_idx, speaker in enumerate(emb_vec):
            cs_row = []
            for utterance_idx, utterance in enumerate(speaker):
                # cosign similarity function described in https://arxiv.org/pdf/1509.08062.pdf
                new_centroids = self.calc_new_centroids(emb_vec, centroid, speaker_idx, utterance_idx)
                cs_row.append(torch.mm(utterance.unsqueeze(1).transpose(0, 1), new_centroids.transpose(0, 1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)


    def scaled_cos_similarity(self, emb_vec, centroid):
        return self.w * self.cos_similarity(emb_vec, centroid) + self.b

    def softmax_loss(self, emb_vec, cos_sim_matrix):
        loss = []

        for i in range(len(emb_vec)):
            row = []
            for j in range(len(emb_vec[0])):
                row.append(-torch.nn.functional.log_softmax(cos_sim_matrix[i,j], 0)[i])
            row = torch.stack(row)
            loss.append(row)

        return torch.stack(loss).sum()



    def forward(self, input):
        #TODO: define forward function to calculate end to end loss.
        #get centroid
        centroids = torch.mean(input, dim=1)
        #get similarity matrix
        cos_sim_matrix = self.scaled_cos_similarity(input, centroids)

        #get loss
        loss = self.softmax_loss(input, cos_sim_matrix)

        return loss
