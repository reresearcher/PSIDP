# encoding: utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from loguru import logger

from scipy.stats import norm
import sklearn.manifold as manifold_tools
from scipy.spatial.distance import pdist, squareform

from model_loader import load_model
from evaluate import mean_average_precision


def train(
          near_neighbor,
          num_train,
          batch_size,
          dataset,
          train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          feature_dim,
          label_dim,
          alpha,
          beta,
          gamma,
          max_iter,
          arch,
          lr,
          device,
          evaluate_interval,
          snapshot_interval,
          topk,
          checkpoint=None,
          ):

    # Model, optimizer, criterion
    model, model0 = load_model(arch, code_length)
    model.to(device)
    model0.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    print('Extract features...')
    attribute_dim = 1000
    features, attributes = extract_features(model, model0, train_dataloader, feature_dim, attribute_dim, device)
    model0.to('cpu')
    
    vgg_features = features.cpu().numpy()
    attributes = attributes.cpu().numpy()

    print('Generate similarity matrix...')
    S_numpy, I_numpy = distance_fusion(vgg_features, alpha, beta, near_neighbor)
    S_numpy = S_numpy.astype(np.int)

    S_buffer = torch.FloatTensor(S_numpy).to(device)
    I_buffer = torch.FloatTensor(I_numpy).to(device)
    attributes_buffer = torch.FloatTensor(attributes).to(device)
    

    I = Variable(I_buffer)
    S = Variable(S_buffer)
    bestMAP = -1.0
    print('Start training...')
    model.train()
    for epoch in range(max_iter):
        n_batch = len(train_dataloader)
        for i, (data, _, index) in enumerate(train_dataloader):

            data = data.to(device)
            index = index.to(device)
            attribute = attributes_buffer[index, :]

            cur_f, cur_att, cur_code = model(data)
            
            v = cur_f
            H = v @ v.t() / code_length
            targets = S[index, :][:, index]
            sim_loss = (targets.abs() * (H - targets).pow(2)).sum() / (H.shape[0] ** 2)


            keyong = I[index, :]
            keyongche = torch.unsqueeze(torch.sum(keyong, dim=1), 0)
            code_loss = torch.sum(torch.mm(keyongche, (torch.pow(cur_code - cur_f, 2))))
            att_loss = torch.sum(torch.mm(keyongche, (torch.pow(cur_att - attribute, 2))))
            
            att_loss = (code_loss+att_loss)/batch_size
            loss = sim_loss + gamma*att_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.debug('[epoch:{}][Batch:{}/{}][loss:{:.4f}]'.format(epoch+1, i+1, n_batch, loss.item()))

        if epoch % evaluate_interval == 0:
            query_code, retrieval_code, onehot_query_targets, onehot_retrieval_targets, mAP = evaluate(model,
                           query_dataloader,
                           retrieval_dataloader,
                           code_length,
                           label_dim,
                           device,
                           topk,
                           )
            
            
            logger.info('[epoch:{}][map:{:.4f}]'.format(epoch, mAP))

            
            
            if mAP > bestMAP:
                bestMAP = mAP
                query_code = query_code.numpy().astype(np.int)
                retrieval_code = retrieval_code.numpy().astype(np.int)
                onehot_query_targets = onehot_query_targets.cpu().numpy().astype(np.int)
                onehot_retrieval_targets = onehot_retrieval_targets.cpu().numpy().astype(np.int)
                
                mat_name = arch + '_' + dataset + '_' + str(code_length) + '_best'

                sio.savemat(mat_name+'.mat', {'Qi':query_code,
                                    'Di':retrieval_code,
                                    'query_L':onehot_query_targets,
                                    'retrieval_L':onehot_retrieval_targets})
        


        # Save snapshot
        if epoch % snapshot_interval == 0:
            checkpoint_name = arch + '_' + dataset + '_' + str(code_length) + '_' + str(max_iter)
            model.snapshot(epoch)
            logger.info('[epoch:{}][Snapshot]'.format(epoch))

def evaluate(model, query_dataloader, retrieval_dataloader, code_length, label_dim, device, topk):

    model.eval()

    # Generate hash code
    print('Generate Query Set Code...')
    query_code, _ = generate_code(model, query_dataloader, code_length, label_dim, device)
    print('Generate Retrieval Set Code...')
    retrieval_code, _ = generate_code(model, retrieval_dataloader, code_length, label_dim, device)
    
    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)


    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    model.train()

    return query_code, retrieval_code, onehot_query_targets, onehot_retrieval_targets, mAP


def generate_code(model, dataloader, code_length, label_dim, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        labels = torch.zeros([N, label_dim])
        for data, label, index in tqdm(dataloader):
            data = data.to(device)
            outputs, _, _ = model(data)
            code[index, :] = outputs.sign().cpu()
            labels[index, :] = label.float()
        labels = labels.to(device)

    return code, labels

def distance_fusion(vgg_features, alpha, beta, near_neighbor):


    print('Seek for VGG embedding...')
    vgg_dist = pdist(vgg_features, metric='cosine')
    vgg_dist = squareform(vgg_dist)

    print('whole field...')
    Svgg = generate_similarity_matrix(vgg_dist, alpha, beta)


    print('near field...')
    LLE = manifold_tools.LocallyLinearEmbedding(n_neighbors=near_neighbor, n_components=512, n_jobs=8)
    lle_embedding = LLE.fit_transform(vgg_features)
    lle_dist = pdist(lle_embedding, metric='cosine')
    lle_dist = squareform(lle_dist)
    Slle = generate_similarity_matrix(lle_dist, alpha, beta)


    print('far field...')
    ISOMAP = manifold_tools.Isomap(n_neighbors=near_neighbor, path_method='FW', n_components=512, n_jobs=8)
    ISOMAP.fit(vgg_features)
    geo_dist = ISOMAP.dist_matrix_
    _range = np.max(geo_dist) - np.min(geo_dist)
    geo_dist = (geo_dist - np.min(geo_dist)) / _range
    Siso = generate_similarity_matrix(geo_dist, alpha, beta)


    print('Flitering...')
    Snear = 1.0 * (Svgg + Slle == 2)
    Sfar = -1.0 * (Svgg + Siso == -2)
    S = Snear + Sfar
    I = (((S == 1.0) + (S == -1.0)) == 1)

    return S, I


def generate_similarity_matrix(dist_matrix, alpha, beta):
    """
    Generate similarity matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = dist_matrix
    # Find maximum count of cosine distance
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval

    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    # Construct similarity matrix
    S = (cos_dist < (left_mean - alpha * left_std)) * 1.0 + (cos_dist > (right_mean + beta * right_std)) * -1.0
    # return torch.FloatTensor(S), torch.FloatTensor(I), torch.FloatTensor(D)
    return S

def extract_features(model, model0, dataloader, feature_dim, attribute_dim, device):

    model.eval()
    model0.eval()
    model.set_extract_features(True)

    features = torch.zeros(len(dataloader.dataset.data), feature_dim)
    attributes = torch.zeros(len(dataloader.dataset.data), attribute_dim)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data, label, index) in enumerate(dataloader):
            logger.debug('[Batch:{}/{}]'.format(i + 1, N))
            data = data.to(device)
            features[index, :] = model(data).cpu()
            attributes[index, :] = model0(data).cpu()

    model.set_extract_features(False)
    model.train()

    return features, attributes
