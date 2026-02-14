import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from extract_embeddings import extract_feats

def resolve_path(*parts):
    return os.path.abspath(os.path.join(BASE_DIR, *parts))


def build_search_index(img_feats, dest_dir, n_train=None, do_pca=False, m=32, nprobe=256, nlist=None, force_flat=False,
                       batch_size=None, use_gpu=False):
    os.makedirs(dest_dir, exist_ok=True)

    if n_train is None:
        train_feats = img_feats
    else:
        train_feats = img_feats[:n_train]
    feat_dim = train_feats.shape[1]

    if len(img_feats) < 100000 or force_flat:
        print('Using flat index')
        index = faiss.IndexFlatIP(feat_dim)
    else:
        print('Using IVFPQ index')
        if nlist is None:
            nlist = int(np.sqrt(train_feats.shape[0]) * 4)
            print('Setting nlist={}'.format(nlist))
        quantizer = faiss.IndexFlatIP(feat_dim)
        index = faiss.IndexIVFPQ(quantizer, feat_dim, nlist, m, 8)
        index.nprobe = nprobe

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    norm1 = faiss.NormalizationTransform(feat_dim)
    norm1.train(train_feats)
    train_feats = norm1.apply_py(train_feats)
    faiss.write_VectorTransform(norm1, os.path.join(dest_dir, 'norm1.bin'))
    if do_pca:
        pca = faiss.PCAMatrix(d_in=feat_dim, d_out=feat_dim, eigen_power=-0.5)
        pca.train(train_feats)
        train_feats = pca.apply_py(train_feats)
        faiss.write_VectorTransform(pca, os.path.join(dest_dir, 'pca.bin'))
        norm2 = faiss.NormalizationTransform(feat_dim)
        norm2.train(train_feats)
        train_feats = norm2.apply_py(train_feats)
        faiss.write_VectorTransform(norm2, os.path.join(dest_dir, 'norm2.bin'))

    print('Training index...')
    index.train(train_feats)

    def transform(x):
        x = norm1.apply_py(x)
        if do_pca:
            x = pca.apply_py(x)
            x = norm2.apply_py(x)
        return x

    if batch_size is None:
        print('Adding vectors to index...')
        feats = transform(img_feats)
        index.add(feats)
    else:
        print('Adding vectors to index in batches...')
        for i in tqdm(range(0, len(img_feats), batch_size)):
            feats = transform(img_feats[i:i+batch_size])
            index.add(feats)
    print('Done adding vectors to index')

    index_path = os.path.join(dest_dir, 'faiss_IVPQ_PCA.index')
    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, index_path)



index_configs = {
    'CLIP-Huge-Flickr-Flat':
        {
            'a1_config': 'ViT-H-14-quickgelu',
            'weight_path': 'dfn5b',
            'img_dir': resolve_path('..', 'coco_data', 'train2017'),
        },
}

out_dir = resolve_path('..', 'coco_faiss_indexes')

if __name__ == '__main__':

    os.makedirs(out_dir, exist_ok=True)
    model_id = 'CLIP-Huge-Flickr-Flat'

    force_flat = 'Flat' in model_id
    save_feats = True

    index_dir = os.path.join(out_dir, model_id)
    os.makedirs(index_dir, exist_ok=True)

    im_hashes, im_feats = extract_feats(index_configs[model_id])

    if save_feats:
        feats_dir = resolve_path('..', 'coco_feats')
        os.makedirs(feats_dir, exist_ok=True)
        np.save(os.path.join(feats_dir, model_id + '_feats.npy'), im_feats)

    np.save(os.path.join(index_dir, 'img_ids.npy'), im_hashes)

    build_search_index(im_feats, index_dir, force_flat=force_flat, use_gpu=True)