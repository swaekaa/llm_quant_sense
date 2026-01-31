def rank_layers_by_sensitivity(results_dict):
    """
    Sort layers by accuracy drop (descending).
    """
    return sorted(
        results_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
