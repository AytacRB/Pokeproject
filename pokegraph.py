import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import ast
from bicm import BipartiteGraph


"""
This script defines the 'Pokegraph()' wrapper function used to construct and load the
sampled network to a graph object.

"""




def Pokegraph(path, directed=False, hr=False, cutoff = 1400, validate=False):

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

    # Preprocess csv file to ensure proper loading

    df = pd.read_csv(path)
    df['players'] = df['players'].apply(ast.literal_eval)
    df['p1'] = df['p1'].apply(ast.literal_eval)
    df['p2'] = df['p2'].apply(ast.literal_eval)
    df = df.drop_duplicates(subset=['id', 'uploadtime'])

    # Optional low-rank filter
    if hr:
        df = df[df['rating'] >= cutoff]
    
    edge_weights = {}
    win_edge_counts = {}
    node_participation_counts = {}
    node_ratings = {}
    win_counts = {}
    participation_counts = {}
    dicts = (edge_weights, win_edge_counts, node_participation_counts, node_ratings, win_counts, participation_counts)

    # Process each row
    for _, row in df.iterrows():
        dicts = update_node_stats(row['p1'], row['p2'], row['rating'], row['winner'], dicts)

    edge_weights, win_edge_counts, node_participation_counts, node_ratings, win_counts, participation_counts = dicts

    # Attribute Generation
    winrates = {node: win_counts[node] / participation_counts[node]
                for node in participation_counts}
    average_ratings = {node: (data['total_rating'] / data['count']) if data['count'] > 0 else 0
                       for node, data in node_ratings.items()}

    playrates = {node: participation_counts[node] / (len(df)*2) 
                for node in participation_counts}

    edge_playrates = {pair: weight / (len(df)*2)
                    for pair, weight in edge_weights.items()}
    


    """
    For undirected networks we generate an empty Graph and fill it from the data dictionaries,
    assigning edges with properties. 


    For directed networks we specify the edge direction when filling the diGraph. 

    """


    if directed == True:
        # Create the graph and compute edge-level attributes
        G = nx.DiGraph()  # Use directed graph to support conditional weight (A → B)
        for (node1, node2), cooccurrence in edge_weights.items():
            wins = win_edge_counts.get((node1, node2), 0)
            conditional_winrate = wins / cooccurrence if cooccurrence > 0 else 0

            # Add conditional weights for both directions
            if node1 in node_participation_counts and node2 in node_participation_counts[node1]:
                conditional_weight_1_to_2 = node_participation_counts[node1][node2] / participation_counts[node1]
                G.add_edge(node1, node2, weight=cooccurrence, 
                        conditional_weight=conditional_weight_1_to_2,
                        conditional_winrate=conditional_winrate)

            if node2 in node_participation_counts and node1 in node_participation_counts[node2]:
                conditional_weight_2_to_1 = node_participation_counts[node2][node1] / participation_counts[node2]
                G.add_edge(node2, node1, weight=cooccurrence, 
                        conditional_weight=conditional_weight_2_to_1,
                        conditional_winrate=conditional_winrate)
    else:
        G = nx.Graph()
        for (node1, node2), cooccurrence in edge_weights.items():
            wins = win_edge_counts.get((node1, node2), 0)
            conditional_winrate = wins / cooccurrence if cooccurrence > 0 else 0
            edge_playrate = edge_playrates.get((node1, node2), 0)
            G.add_edge(node1, node2,
             weight=cooccurrence,
             conditional_winrate = conditional_winrate,
             edge_playrate = edge_playrate)

    # Adding node attributes
    nx.set_node_attributes(G, winrates, 'winrate')
    nx.set_node_attributes(G, average_ratings, 'rating_avg')
    nx.set_node_attributes(G, playrates, 'playrate')    
    node_data = (winrates, average_ratings, playrates)


    #  BiCM validation call and filtering.
    if validate:
        G_final = nx.Graph()
        G_val = validated_pokegraph(path,hr,cutoff)

        for u, v in G_val.edges():
            G_final.add_edge(u, v, **G[u][v])

        for node in G_final.nodes():
            G_final.nodes[node].update(G.nodes[node])

        
        G = G_final
        winrates, average_ratings, playrates = node_data
        filtered_node_data = (
            {n: winrates[n] for n in G_final.nodes if n in winrates},
            {n: average_ratings[n] for n in G_final.nodes if n in average_ratings},
            {n: playrates[n] for n in G_final.nodes if n in playrates}
            )
        node_data = filtered_node_data





    return (G, df, node_data)


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
                        node_participation_counts[source][target] = (
                            node_participation_counts[source].get(target, 0) + 1
                        )
        dicts = (edge_weights, win_edge_counts, node_participation_counts, node_ratings, win_counts, participation_counts)


        return dicts

def validated_pokegraph(path, hr, cutoff):

    """
    BiCM - Application function. Construct Bipartite Graph from data, apply BiCM workflow.

    Args:
        path: String reference to input data
    
    Returns:
            validated_G: G after BiCM filtering. 
    
    """

    df = pd.read_csv(path)
    df['players'] = df['players'].apply(ast.literal_eval)
    df['p1'] = df['p1'].apply(ast.literal_eval)
    df['p2'] = df['p2'].apply(ast.literal_eval)
    df = df.drop_duplicates(subset=['id', 'uploadtime'])
    if hr:
        df = df[df['rating'] >= cutoff]

    G = nx.Graph()

    team_counter = 1  # Unique team counter

    for _, row in df.iterrows():
        for col in ['p1', 'p2']: 
            team_units = row[col]  
            team_name = f"team_{team_counter}"
            G.add_node(team_name, bipartite=0)  # Team node
            
            for unit in team_units:
                G.add_node(unit, bipartite=1)  # Unit node
                G.add_edge(team_name, unit)  # Connect team to its units
            
            team_counter += 1  

    # Split nodes
    teams = sorted({n for n, d in G.nodes(data=True) if d['bipartite'] == 0})
    units = sorted({n for n, d in G.nodes(data=True) if d['bipartite'] == 1})

    team_index = {team: i for i, team in enumerate(teams)}
    unit_index = {unit: i for i, unit in enumerate(units)}

    # build matrix
    biadj_matrix = np.zeros((len(teams), len(units)), dtype=int)

    for team in teams:
        for unit in G.neighbors(team):
            biadj_matrix[team_index[team], unit_index[unit]] = 1

    # main BiCM computation
    BG = BipartiteGraph(biadjacency=biadj_matrix)
    BG.solve_tool()
    BG.compute_projection(rows=False, alpha=0.01, approx_method='poisson', progress_bar=True)

    # Map unit indices back to names
    index_to_unit = {i: unit for unit, i in unit_index.items()}  # invert original mapping
    unit_projection = BG.get_cols_projection(fmt='edgelist')
    # Translate edges to names
    named_edges = [(index_to_unit[u], index_to_unit[v]) for u, v in unit_projection]

    # Build named graph
    validated_G = nx.Graph()
    validated_G.add_edges_from(named_edges)    

    return validated_G