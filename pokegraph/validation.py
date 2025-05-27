import networkx as nx
import numpy as np
from bicm import BipartiteGraph
from data_loader import load_and_preprocess_data

def validated_pokegraph(path, hr, cutoff):
    """
    BiCM - Application function. Construct Bipartite Graph from data, apply BiCM workflow.

    Args:
        path: String reference to input data
    
    Returns:
            validated_G: G after BiCM filtering. 
              """
    df = load_and_preprocess_data(path, hr, cutoff)

    G = nx.Graph()
    team_counter = 1    # Unique team counter

    for _, row in df.iterrows():
        for col in ['p1', 'p2']:
            team_units = row[col]
            team_name = f"team_{team_counter}"
            G.add_node(team_name, bipartite=0) # Team node

            for unit in team_units:
                G.add_node(unit, bipartite=1) # Unit node
                G.add_edge(team_name, unit) # Connect team to its units
            team_counter += 1
        
    # Split nodes

    teams = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    units = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

    # build matrix
    biadj_matrix = np.zeros((len(teams), len(units)), dtype=int)
    team_idx = {team: i for i, team in enumerate(teams)}
    unit_idx = {unit: i for i, unit in enumerate(units)}

    for team in teams:
        for unit in G.neighbors(team):
            biadj_matrix[team_idx[team], unit_idx[unit]] = 1

    # main BiCM computation

    BG = BipartiteGraph(biadjacency=biadj_matrix)
    BG.solve_tool()
    BG.compute_projection(rows=False, alpha=0.01, approx_method='poisson', progress_bar=True)
    edges = BG.get_cols_projection(fmt='edgelist')

    # Map unit indices back to names
    idx_to_unit = {i: u for u, i in unit_idx.items()} # invert original mapping
    # Translate edges to names
    named_edges = [(idx_to_unit[u], idx_to_unit[v]) for u, v in edges]
    
    # Build named graph

    validated_G = nx.Graph()
    validated_G.add_edges_from(named_edges)
    return validated_G
