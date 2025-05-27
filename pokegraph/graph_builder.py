import networkx as nx

"""
    For undirected networks we generate an empty Graph and fill it from the data dictionaries,
    assigning edges with properties. 


    For directed networks we specify the edge direction when filling the diGraph. 

"""
def build_graph(edge_weights, win_edge_counts, node_participation_counts,
                winrates, average_ratings, playrates, participation_counts, 
                directed, total_battles):

    edge_playrates = {pair: weight / (total_battles * 2) for pair, weight in edge_weights.items()}

    if directed:
        # Create the graph and compute edge-level attributes
        G = nx.DiGraph()  # Use directed graph to support conditional weight (A → B)
        for (node1, node2), cooccurrence in edge_weights.items():
            wins = win_edge_counts.get((node1, node2), 0)
            conditional_winrate = wins / cooccurrence if cooccurrence > 0 else 0

             # Add conditional weights for both directions
            if node1 in node_participation_counts and node2 in node_participation_counts[node1]:
                weight_1_2 = node_participation_counts[node1][node2] / participation_counts[node1]
                G.add_edge(node1, node2, weight=cooccurrence,
                           conditional_weight=weight_1_2,
                           conditional_winrate=conditional_winrate)

            if node2 in node_participation_counts and node1 in node_participation_counts[node2]:
                weight_2_1 = node_participation_counts[node2][node1] / participation_counts[node2]
                G.add_edge(node2, node1, weight=cooccurrence,
                           conditional_weight=weight_2_1,
                           conditional_winrate=conditional_winrate)
    else:
        G = nx.Graph()
        for (node1, node2), cooccurrence in edge_weights.items():
            conditional_winrate = win_edge_counts.get((node1, node2), 0) / cooccurrence if cooccurrence > 0 else 0
            edge_playrate = edge_playrates.get((node1, node2), 0)
            G.add_edge(node1, node2,
                       weight=cooccurrence,
                       conditional_winrate=conditional_winrate,
                       edge_playrate=edge_playrate)
    
    # Adding node attributes

    nx.set_node_attributes(G, winrates, 'winrate')
    nx.set_node_attributes(G, average_ratings, 'rating_avg')
    nx.set_node_attributes(G, playrates, 'playrate')

    return G
