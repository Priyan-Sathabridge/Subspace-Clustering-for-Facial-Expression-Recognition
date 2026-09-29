import pandas as pd


def load_data(gt_csv, cluster_csv, image_col):
    """Join identical image sets one-to-one; reject silent sample loss."""
    gt = pd.read_csv(gt_csv)
    clusters = pd.read_csv(cluster_csv)
    for name, frame in [('ground truth', gt), ('cluster labels', clusters)]:
        if image_col not in frame or frame.empty:
            raise ValueError(f'{name}: missing image identifiers or empty data')
        if frame[image_col].isna().any() or frame[image_col].duplicated().any():
            raise ValueError(f'{name}: null or duplicate image identifiers')
    if set(gt[image_col]) != set(clusters[image_col]):
        raise ValueError('Ground truth and cluster labels must contain exactly the same images')
    merged = gt.merge(clusters, on=image_col, how='inner', validate='one_to_one')
    if merged[['Label', 'Cluster']].isna().any().any():
        raise ValueError('Missing ground-truth or cluster labels')
    return merged
