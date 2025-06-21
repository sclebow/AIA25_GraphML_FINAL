import networkx as nx
import os
import plotly.graph_objects as go
import pandas as pd
import random
import numpy as np

from neo4j import GraphDatabase
from tqdm import tqdm
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import streamlit as st

# Connect to Neo4j
URI = "bolt://localhost:7687"

USERNAME = "neo4j"
PASSWORD = "macad2025"
DATABASE = 'wbsgraph'

def build_wbs_graph(df_elements, key_path_nodes=None):
    
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
                    G.add_edge(prev_node, curr_node, time=time_of_work, inv_time=1./time_of_work)

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
    color_map = {label: f"rgb({random.randint(50, 200)},{random.randint(50, 200)},{random.randint(50, 200)})" for label in unique_labels}

    work_hours = [G.nodes[n].get('total_work_hours', 0) for n in G.nodes]
    min_hours = min(work_hours)
    max_hours = max(work_hours)
    norm = mcolors.Normalize(vmin=min_hours, vmax=max_hours)
    cmap = plt.get_cmap('jet')
    edge_traces = []
    for source, target in G.edges():
        x0, y0, z0 = pos_3d[source]
        x1, y1, z1 = pos_3d[target]
        edge_color = cmap(norm(G.nodes[source]['total_work_hours']))
        rgb = tuple(int(255 * x) for x in edge_color[:3])
        edge_traces.append(go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(color=f'rgb{rgb}', width=2),
            text="time {} hr".format(G.nodes[source]['total_work_hours'],
            hoverinfo='text')
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
        title="Construction sequence dependency graph - work zone 1",
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )

    graph_fig.write_html("./plots/wbs_graph.html")

    edges = pd.DataFrame(G.edges(data=True), columns=['source', 'target', 'attributes'])

    return graph_fig, edges, G

def load_to_neo4j(G, neo4j_input_dir, reset=False):
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
            # Instead of dropping/creating the database, clear all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully!")
    
    nodes_01 = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')


    def replace_nans(df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df.fillna({col: 0}, inplace=True)
            else:
                df.fillna({col: 'N/A'}, inplace=True)
        return df

    nodes_01 = replace_nans(nodes_01)

    with st.expander("Nodes DataFrame before processing"):
        st.dataframe(nodes_01)

    # nodes dataframe headers: wbs,label,level,work_zone,total_work_hours,LocationX,LocationY,LocationZ
    # Add a 'GlobalId' column that is the index
    nodes_01['GlobalId'] = nodes_01.index
    # Add a 'Name' column that is the wbs of the node
    nodes_01['Name'] = nodes_01.apply(lambda row: row['wbs'] if 'wbs' in row else 'Unknown', axis=1)
    # Add a 'Description' column that is all None
    nodes_01['Description'] = None
    # Add a 'ObjectType' column that is None
    nodes_01['ObjectType'] = None
    # Add a 'IfcType' column that is the label of the node
    nodes_01['IfcType'] = nodes_01.apply(lambda row: row['label'] if 'label' in row else 'Unknown', axis=1)
    # Add a 'category' column that is the label of the node
    nodes_01['category'] = nodes_01.apply(lambda row: row['label'] if 'label' in row else 'Unknown', axis=1)


    # Drop all other columns except for GlobalId, Name, Description, ObjectType, IfcType, category
    columns_to_keep = ['GlobalId', 'Name', 'Description', 'ObjectType', 'IfcType', 'category']
    nodes_01 = nodes_01[columns_to_keep]
    
    with st.expander("Nodes DataFrame after processing"):
        st.dataframe(nodes_01)

    edges_01 = pd.DataFrame(G.edges(data=True), columns=['source', 'target', 'attributes'])
    with st.expander("Edges DataFrame before processing"):
        st.dataframe(edges_01)
    # Check for edges types
    edges_01['relation_type'] = 'RELATED_TO'  # Default relation type

    with st.expander("Edges DataFrame after adding relation_type"):
        st.dataframe(edges_01)

    # Save nodes and edges to CSV files
    nodes_01.to_csv("./data/nodes_01.csv", index=False)
    edges_01.to_csv("./data/edges_01.csv", index=False)

    # Save nodes and edges to Neo4j input directory
    nodes_01.to_csv(os.path.join(neo4j_input_dir, "nodes_01.csv"), index=False)
    edges_01.to_csv(os.path.join(neo4j_input_dir, "edges_01.csv"), index=False)

    # Batch size
    batch_size = 500
    with driver.session(database=DATABASE) as session:
        for i in tqdm(range(0, len(nodes_01), batch_size), desc="Batch merging nodes"):
            batch = nodes_01.iloc[i:i+batch_size].to_dict('records')
            session.execute_write(batch_merge_nodes, batch)

    driver.close()
    print("Nodes loaded successfully!")

    edges_data = []
    for _, row in edges_01.iterrows():
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
            'relation_type': row['relation_type'],
            'props': props
        })

    grouped_edges = defaultdict(list)
    for row in edges_data:
        grouped_edges[row['relation_type']].append(row)

    with driver.session(database=DATABASE) as session:
        for relation_type, group in grouped_edges.items():
            for i in tqdm(range(0, len(group), batch_size), desc=f"Merging {relation_type}"):
                batch = group[i:i+batch_size]
                session.execute_write(batch_merge_edges_without_apoc, relation_type, batch)

    driver.close()
    print("Edges loaded successfully!")

def run_cypher(query, params=None, write=False):
    """
    Executes a Cypher query on the Neo4j database.

    Parameters:
    - query (str): The Cypher query to be executed.
    - params (dict, optional): A dictionary of parameters for the query.
    - write (bool, optional): Set to True if this is a write transaction, otherwise False (default).

    Returns:
    - list[dict]: The query result as a list of dictionaries.
    """
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    if params is None:
        params = {}

    # Open a new session with the correct database
    with driver.session(database=DATABASE) as session:
        # Depending on the type of transaction, use read or write
        if write:
            result = session.write_transaction(lambda tx: tx.run(query, **params).data())
        else:
            result = session.execute_read(lambda tx: tx.run(query, **params).data())
    return result

def build_gds_graph():
    # Check if 'myGraph' exists, if so - drop it
    query_check = """
    CALL gds.graph.exists('myGraph')
    YIELD exists
    """

    result_data = run_cypher(query_check)

    # If the graph exists, drop it
    if result_data and result_data[0]['exists']:
        query_drop = """
        CALL gds.graph.drop('myGraph')
        """
        run_cypher(query_drop)
        print("Existing graph 'myGraph' was dropped successfully.")

    # Now create the graph projection
    # ---------------------------------------------------------------------
    # Properties could be only NUM or BOOL

    query_create = """
    CALL gds.graph.project(
    'myGraph',
    {
        A1034_005_Elevator_Pit_Wall_ID: { properties: [] },
        B1012_HSS_C_060_HSS_Steel_Column_12x12x3_8: { properties: [] },
        B2011_150_Rainscreen__Wall__ID: { properties: [] },
        B1011_026_CIP_RC_Elevator_Core_Wall_ID: { properties: []}
    },
    {
        RELATED_TO: { orientation: 'UNDIRECTED', properties: ['inv_time', 'time']  }
    }
    )

    """ #, properties: 'inv_time

    result_data = run_cypher(query_create, write=True)
    node_query = """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN ['A1034_005_Elevator_Pit_Wall_ID','B1012_HSS_C_060_HSS_Steel_Column_12x12x3_8', 'B2011_150_Rainscreen__Wall__ID', 'B1011_026_CIP_RC_Elevator_Core_Wall_ID'])
        RETURN
        id(n) AS id,
        labels(n)[0] AS label,
        n.GlobalId AS Id
    """
    edge_query = """
        MATCH (a)-[r]->(b)
        WHERE type(r) IN ['RELATED_TO']
        AND any(lbl IN labels(a) WHERE lbl IN ['A1034_005_Elevator_Pit_Wall_ID','B1012_HSS_C_060_HSS_Steel_Column_12x12x3_8', 'B2011_150_Rainscreen__Wall__ID', 'B1011_026_CIP_RC_Elevator_Core_Wall_ID'])
        AND any(lbl IN labels(b) WHERE lbl IN ['A1034_005_Elevator_Pit_Wall_ID','B1012_HSS_C_060_HSS_Steel_Column_12x12x3_8', 'B2011_150_Rainscreen__Wall__ID', 'B1011_026_CIP_RC_Elevator_Core_Wall_ID'])
        RETURN id(a) AS source, id(b) AS target, type(r) AS relationshipType
        """

    edges = run_cypher(edge_query)
    print(f'Number of edges retrieved: {len(edges)}')

    nodes = run_cypher(node_query)
    print(f'Number of nodes retrieved: {len(nodes)}')

    # query_inv_time_edge = """MATCH (source:Location)-[r:ROAD]->(target:Location)
    # RETURN gds.graph.project(
    # 'myGraph',
    # source,
    # target,
    # { relationshipProperties: r { .inv_time } }
    # )"""
    # result_data = run_cypher(query_create, write=True)
    print("Graph 'myGraph' created successfully!")

def shortest_path(start_id, end_id):
    query = f"""
    MATCH (start {{GlobalId: {start_id}}}),
        (end {{GlobalId: {end_id}}})
    CALL gds.shortestPath.dijkstra.stream('myGraph', {{
        sourceNode: id(start),
        targetNode: id(end),
        relationshipWeightProperty: 'inv_time'
    }})
    YIELD index, nodeIds, costs, totalCost
    UNWIND nodeIds AS nodeId
    WITH index, costs, totalCost, gds.util.asNode(nodeId).GlobalId AS GlobalId
    RETURN index, costs, totalCost, collect(GlobalId) AS GlobalIdPath

    """#relationshipWeightProperty: 'inv_time'

    result_data = run_cypher(query)
    print(result_data)
    return result_data

def batch_merge_edges_without_apoc(tx, relation_type, batch):
    query = f"""
    UNWIND $rows AS row
    MATCH (a {{GlobalId: row.source}})
    MATCH (b {{GlobalId: row.target}})
    MERGE (a)-[r:{relation_type}]->(b)
    SET r += row.props
    """
    tx.run(query, rows=batch)

def batch_merge_nodes(tx, batch):
    """
    Merges a batch of nodes into Neo4j with dynamic labels from 'IfcType'
    """
    bad_characters = [" ","-","(",")",".","/"]
    label_groups = defaultdict(list)
    for row in batch:
        label = row.get("Name")
        for verboten in bad_characters:
            label = label.replace(verboten,"_")
        if label:
            label_groups[label].append(row)

    for label, records in label_groups.items():
        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{GlobalId: row.GlobalId}})
        SET n += row
        """
        tx.run(query, rows=records)
    # """
    # Merges a batch of nodes into Neo4j using GlobalId for matching and updates all properties.

    # Uses UNWIND to process each node in the batch.
    # """
    # query = """
    # UNWIND $rows AS row
    # MERGE (n:YourLabel {GlobalId: row.GlobalId})
    # SET n += row
    # """  # Removed the comment inside the query string
    # tx.run(query, rows=batch)

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