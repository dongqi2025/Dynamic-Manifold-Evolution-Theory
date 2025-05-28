import torch


def _corr(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def analyze_correlation(dynamics, quality):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = torch.stack([
        torch.tensor([d['mean_distance'] for d in dynamics]),
        torch.tensor([d['silhouette_score'] for d in dynamics]),
        1 / (torch.tensor([d['final_loss'] for d in dynamics]) + 1e-6),
        torch.tensor([d['H1_persistence'] for d in dynamics])
    ], dim=1).to(device)

    targets = torch.stack([
        torch.log(torch.tensor(quality['perplexity'])),
        torch.tensor(quality['spelling']),
        torch.tensor(quality['diversity']),
        torch.tensor(quality['grammar']),
        torch.tensor(quality['coherence'])
    ], dim=1).to(device)

    results = {}
    for i, target_name in enumerate(['log_perplexity', 'spelling', 'diversity', 'grammar', 'coherence']):
        corrs = []
        for j in range(features.size(1)):
            corrs.append(_corr(features[:, j], targets[:, i]).item())
        results[target_name] = corrs

    return results
