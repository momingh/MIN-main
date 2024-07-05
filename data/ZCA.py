import pdb

import torch
import numpy as np

def zca_whitening(X, epsilon=1e-5):
    X_centered = X - torch.mean(X, axis=0)
    sigma = torch.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    whitening_matrix = torch.matmul(
        eigenvectors,
        torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
    ).matmul(eigenvectors.T)
    X_whitened = torch.matmul(X_centered, whitening_matrix)
    return X_whitened


def group_zca_whitening(X, groups):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    unique_groups = torch.unique(groups)
    X_whitened = torch.zeros_like(X)

    for group in unique_groups:
        group_mask = (groups == group)
        X_group = X[group_mask]
        X_whitened_group = zca_whitening(X_group)
        X_whitened[group_mask] = X_whitened_group

    return X_whitened


示例用法
if __name__ == "__main__":
    # 示例数据矩阵 X 和分组标签

    dataset = "baby"
    images = np.load(f'{dataset}/image_feat.npy')
    texts = np.load(f'{dataset}/text_feat.npy')

    num_samples = images.shape[0]
    groups = torch.randint(0, 1, (num_samples,))
    I_whitened = group_zca_whitening(images, groups)
    T_whitened = group_zca_whitening(texts, groups)

    np.save(f'{dataset}/image_feat_whitened.npy', I_whitened)
    np.save(f'{dataset}/text_feat_whitened.npy', T_whitened)
