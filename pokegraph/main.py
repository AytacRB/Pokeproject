from data_loader import load_and_preprocess_data
from stats import update_node_stats, compute_node_attributes
from graph_builder import build_graph
from validation import validated_pokegraph
import networkx as nx

def Pokegraph(path, directed=False, hr=False, cutoff=1400, validate=False):
    """
    Main function interface. Wraps graph construction, validation and statistic computation.

    Args:
        path: String reference to input data
        directed: True generates directed edge-attributes
        hr: True Limits sample to high-ranking battles
        cutoff: specifies high-rank cutoff

    Returns:
        G: networkx Graph object constructed from input data
        df: processed tabular data
        node_data: Tuple containing node-level statistics (winrate / rating_avg / playrate)

    """
    df = load_and_preprocess_data(path, hr, cutoff)
    dicts = ({}, {}, {}, {}, {}, {})
    for _, row in df.iterrows():
        dicts = update_node_stats(row['p1'], row['p2'], row['rating'], row['winner'], dicts)

    edge_weights, win_edge_counts, node_participation_counts, node_ratings, win_counts, participation_counts = dicts
    winrates, average_ratings, playrates = compute_node_attributes(win_counts, participation_counts, node_ratings, len(df))
    
    G = build_graph(edge_weights, win_edge_counts, node_participation_counts, winrates,
                    average_ratings, playrates, participation_counts, directed, len(df))
    
    node_data = (winrates, average_ratings, playrates)

    if validate:
        G_val = validated_pokegraph(path, hr, cutoff)
        G_final = nx.Graph()
        for u, v in G_val.edges():
            G_final.add_edge(u, v, **G[u][v])
        for n in G_final.nodes():
            G_final.nodes[n].update(G.nodes[n])
        G = G_final
        node_data = tuple({k: v for k, v in d.items() if k in G.nodes} for d in node_data)

    return G, df, node_data
