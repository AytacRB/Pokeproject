from itertools import combinations
import pandas as pd

def update_node_stats(p1_list, p2_list, rating, winner, dicts):

    """
    Iteratively called on each row to update the above dictionaries by rowwise
    processing of battles. Yields dictionaries used for final graph construction. 

    Args: 
        p1_list: player-1 team list
        p2_list: player-2 team list
        rating: combined player rating
        winner: winner of battle (either p1 or p2)
        dicts: Tuple of data dictionaries

    Returns:
        dicts: Updated dictionaries
    """
    edge_weights, win_edge_counts, node_participation_counts, node_ratings, win_counts, participation_counts = dicts
    all_nodes = set(p1_list + p2_list)

    # Track participation and wins for individual nodes
    for node in all_nodes:
        participation_counts[node] = participation_counts.get(node, 0) + 1
        win_counts[node] = win_counts.get(node, 0)

        # Initialize node ratings if not already present
        if node not in node_ratings:
            node_ratings[node] = {'total_rating': 0, 'count': 0}

    if winner == 1:
        for node in p1_list:
            win_counts[node] += 1
    elif winner == 2:
        for node in p2_list:
            win_counts[node] += 1

    # Update node ratings
    for node in all_nodes:
        if not pd.isnull(rating):
            node_ratings[node]['total_rating'] += rating
            node_ratings[node]['count'] += 1

    # Update edge stats
    for lst, win in zip([p1_list, p2_list], [winner == 1, winner == 2]):
        for pair in combinations(lst, 2):
            pair = tuple(sorted(pair))
            edge_weights[pair] = edge_weights.get(pair, 0) + 1
            if win:
                win_edge_counts[pair] = win_edge_counts.get(pair, 0) + 1
        # Track participation counts for directional edges
        for source in lst:
            if source not in node_participation_counts:
                node_participation_counts[source] = {}
            for target in lst:
                if source != target:
                    node_participation_counts[source][target] = node_participation_counts[source].get(target, 0) + 1

    return (edge_weights, win_edge_counts, node_participation_counts, node_ratings, win_counts, participation_counts)

def compute_node_attributes(win_counts, participation_counts, node_ratings, total_battles):
    """
    Generate node attribute dictionaries
    """

    winrates = {node: win_counts[node] / participation_counts[node]
                for node in participation_counts}
    average_ratings = {node: (data['total_rating'] / data['count']) if data['count'] > 0 else 0
                       for node, data in node_ratings.items()}
    playrates = {node: participation_counts[node] / (total_battles * 2)
                 for node in participation_counts}
    return winrates, average_ratings, playrates
