import networkx as nx
import os
import plotly.graph_objects as go
import pandas as pd
import random
import numpy as np

from neo4j import GraphDatabase
from tqdm import tqdm
from collections import defaultdict

# Connect to Neo4j
URI = "bolt://localhost:7687"

USERNAME = "neo4j"
PASSWORD = "macad2025"
DATABASE = 'neo4j'

def build_wbs_graph(df_elements):
    
    work_zones = df_elements['work_zone'].unique()

    levels = df_elements['level'].unique()
    G = nx.DiGraph()
    for workey_zoney in work_zones:
        workzone_elements = df_elements[df_elements['work_zone']==workey_zoney]
        node_lists = []
        for level in workzone_elements['level'].unique():
            level_elements = workzone_elements[workzone_elements['level']==level]
            
            for name in level_elements['name'].unique():
                name_elements = level_elements[level_elements['name'] == name]
                node_list = []
                for _, element in name_elements.iterrows():
                    G.add_node(
                        element['id'], 
                        # node_index,
                        wbs=element['name'], 
                        label=element['type'],
                        level=element['level'], 
                        work_zone=element['work_zone'], 
                        total_work_hours=element.get('total_work_hours', 0), 
                        LocationX=element['location']['x'],
                        LocationY=element['location']['y'],
                        LocationZ=element['location']['z'])
                    node_list.append(element['id'])

                node_lists.append(node_list)

        # Build edges between lists of nodes
        for index, node_list in enumerate(node_lists):
            if index == 0:
                continue

            previous_node_list = node_lists[index - 1]
            # nodes, node_times = G.nodes(data='total_work_hours')
            # Create edges between all nodes in the previous list and the current list
            for prev_node in previous_node_list:
                for curr_node in node_list:
                    time_of_work = G.nodes[curr_node]['total_work_hours']
                    G.add_edge(prev_node, curr_node, time=time_of_work)

    # Remove blank nodes
    blank_nodes = [node for node, data in G.nodes(data=True) if 'label' not in data]
    G.remove_nodes_from(blank_nodes)
    # Assign positions from node attributes
    pos_3d = {
        node: (
            G.nodes[node].get("LocationX", 0),
            G.nodes[node].get("LocationY", 0),
            G.nodes[node].get("LocationZ", 0)
        )
        for node in G.nodes
    }

    # for node, data in G.nodes(data='position'):
    #     # Convert position to a numpy array for easier manipulation
    #     if data:
    #         # print(f"Node: {node}, Position: {data}")
    #         # print(f"Type: {type(data)}")
    #         pos_3d[node] = np.array([data['x'], data['y'], data['z']])
    #     else:
    #         pos_3d[node] = np.array([0, 0, 0])  # Default position if no data is available

    unique_labels = df_elements['name'].unique()
    color_map = {label: f"rgb({random.randint(100, 255)},{random.randint(100, 255)},{random.randint(100, 255)})" for label in unique_labels}

    edge_traces = []
    for source, target in G.edges():
        x0, y0, z0 = pos_3d[source]
        x1, y1, z1 = pos_3d[target]
        edge_traces.append(go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none'
        ))
    
    node_trace = go.Scatter3d(
        x = [pos_3d[node][0] for node in G.nodes],
        y = [pos_3d[node][1] for node in G.nodes],
        z = [pos_3d[node][2] for node in G.nodes],
        mode='markers',
        text=[f"Label: {data}<br>" for node, data in G.nodes(data=True)],
        hoverinfo='text',
        marker=dict(
            size=5,
            color=[color_map.get(data['wbs'], 'lightgray') if 'label' in data else 'red' for node, data in G.nodes(data=True)],
            opacity=0.9
        )
    )

    graph_fig = go.Figure(data=[node_trace] + edge_traces)
    graph_fig.update_layout(
        title="3D Spring Layout of IFC Graph",
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )

    graph_fig.write_html("./plots/graph.html")

    edges = pd.DataFrame(G.edges(data=True), columns=['source', 'target', 'attributes'])

    load_to_neo4j(G)

    return graph_fig, edges

def load_to_neo4j(G, reset=False):
    # Create a Neo4j driver
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    # Verify connectivity
    try:
        driver.verify_connectivity()
        print("Connection successful!")
    except Exception as e:
        print("Connection failed:", e)
    if reset:
        with driver.session(database=DATABASE) as session:
        # Reset the database
            session.run("DROP DATABASE neo4j IF EXISTS")
            session.run("SHOW DATABASES")
            print("Database reset successfully!")
    
    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

    def replace_nans(df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df.fillna({col: 0}, inplace=True)
            else:
                df.fillna({col: 'N/A'}, inplace=True)
        return df

    nodes = replace_nans(nodes)

    edges = pd.DataFrame(G.edges(data=True), columns=['source', 'target', 'attributes'])
    # Check for edges types
    edges['relation_type'] = edges['attributes'].apply(lambda x: x.get('relation', None))

    batch_size = 500
    with driver.session(database=DATABASE) as session:
        for i in tqdm(range(0, len(nodes), batch_size), desc="Batch merging nodes"):
            batch = nodes.iloc[i:i+batch_size].to_dict('records')
            session.execute_write(batch_merge_nodes, batch)
    print("Nodes loaded successfully!")

    edges_data = []
    for _, row in edges.iterrows():
        props = {}
        # Flatten the 'attributes' dictionary if present
        if isinstance(row['attributes'], dict):
            props.update(row['attributes'])
        # Add other columns as properties, except for source, target, relation_type, attributes
        for col in row.index:
            if col not in ['source', 'target', 'relation_type', 'attributes']:
                props[col] = row[col]
        edges_data.append({
            'source': row['source'],
            'target': row['target'],
            'props': props
        })

    # Batch size
    batch_size = 500
    with driver.session(database=DATABASE) as session:
        for i in tqdm(range(0, len(edges_data), batch_size), desc="Merging edges"):
            batch = edges_data[i:i + batch_size]
            session.execute_write(batch_merge_edges, batch)

    print("Edges loaded successfully!")



def batch_merge_nodes(tx, batch):
    """
    Merges a batch of nodes into Neo4j with dynamic labels from 'IfcType'
    """

    label_groups = defaultdict(list)
    for row in batch:
        label = row.get("IfcType")
        if label:
            label_groups[label].append(row)

    for label, records in label_groups.items():
        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{GlobalId: row.GlobalId}})
        SET n += row
        """
        tx.run(query, rows=records)

def batch_merge_edges(tx, batch):
    query = """
    UNWIND $rows AS row
    MATCH (a {GlobalId: row.source})
    MATCH (b {GlobalId: row.target})
    MERGE (a)-[r:RELATED_TO]->(b)
    SET r += row.props
    """
    tx.run(query, rows=batch)

# Retrieve data from Neo4j database
def get_nodes(tx):
    """
    Retrieve all nodes with all their properties.
    """
    query = "MATCH (n) RETURN properties(n) AS props"
    return [record["props"] for record in tx.run(query)]

def get_edges(tx):
    """
    Retrieve all relationships (edges) between nodes using GlobalId.
    """
    query = "MATCH (a)-[r]->(b) RETURN a.GlobalId AS source, b.GlobalId AS target"
    return [{"source": record["source"], "target": record["target"]} for record in tx.run(query)]