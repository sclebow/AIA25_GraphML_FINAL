import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd
import random
from main import load_ifc_file, create_all_elements_dict, assign_levels, assign_work_zones
from dependency_utils import get_wbs_from_directory, load_wbs
import numpy as np


st.set_page_config(page_title="IFC Level & Work Zone Visualizer", layout="wide")
st.title("IFC Level & Work Zone Visualizer")

# File selection from ./ifc directory
ifc_dir = "./ifc"
if not os.path.exists(ifc_dir):
    st.error(f"IFC directory '{ifc_dir}' not found.")
    st.stop()

# ifc_files = [f for f in os.listdir(ifc_dir) if f.endswith('.ifc')]
# if not ifc_files:
#     st.error("No IFC files found in the './ifc' directory.")
#     st.stop()

# selected_ifc = st.selectbox("Select IFC file", ifc_files)
# ifc_path = os.path.join(ifc_dir, selected_ifc)
ifc_files = [f for f in os.listdir(ifc_dir) if f.endswith('.ifc')]
if not ifc_files:
    st.error("No IFC files found in the './ifc' directory.")
    st.stop()

latest_ifc_index = max(range(len(ifc_files)), key=lambda i: os.path.getmtime(os.path.join(ifc_dir, ifc_files[i])))
ifc_files.sort(key=lambda f: os.path.getmtime(os.path.join(ifc_dir, f)), reverse=False)
st.write(f"Latest IFC file: {ifc_files[latest_ifc_index]}")

selected_ifc = st.selectbox("Select IFC file", ifc_files, index=latest_ifc_index)
ifc_path = os.path.join(ifc_dir, selected_ifc)

with st.spinner("Loading IFC file and processing..."):
    ifc_file = load_ifc_file(ifc_path)
    if ifc_file is None:
        st.error("Failed to load IFC file.")
        st.stop()

with st.spinner("Loading IFC file and processing..."):
    ifc_file = load_ifc_file(ifc_path)
    if ifc_file is None:
        st.error("Failed to load IFC file.")
        st.stop()

if ifc_file:
    all_elements_dict = create_all_elements_dict(ifc_file)

    # Filter elements by Name
    # Name does not start with 'LVL' or '1000'
    df_elements = pd.DataFrame.from_dict(all_elements_dict, orient='index')
    df_elements = df_elements[~df_elements['name'].str.startswith(('LVL', '1000', 'Stair'), na=False)]
    df_elements = df_elements[df_elements['name'].notna() & (df_elements['name'] != '')]
    all_elements_dict = df_elements.to_dict(orient='index')

    # Display a table of elements
    with st.expander("Elements Overview"):
        st.dataframe(pd.DataFrame.from_dict(all_elements_dict, orient='index'))
    with st.expander("View unique element names", expanded=False):
        unique_names = sorted(set(e['name'] for e in all_elements_dict.values() if e['name']))
        st.table(pd.DataFrame(unique_names, columns=["Unique Element Names"]))
    
    st.markdown("---")
    st.markdown("### Assign Levels")
    level_threshold = st.slider("Level clustering threshold (Z, meters)", min_value=0.01, max_value=2.0, value=0.1, step=0.01)
    all_elements_dict, levels_fig = assign_levels(all_elements_dict, ifc_file, threshold=level_threshold, plot=True)
    st.success("Levels assigned.")
    unique_levels = sorted(set(e['level'] for e in all_elements_dict.values() if e['level'] != -1))
    st.write(f"**Number of unique levels found:** {len(unique_levels)}")
    st.subheader("Levels Plot")
    if levels_fig:
        st.plotly_chart(levels_fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Assign Work Zones")
    number_of_zones = st.slider("Number of work zones to assign", min_value=1, max_value=20, value=6, step=1)
    all_elements_dict, work_zones_fig = assign_work_zones(all_elements_dict, ifc_file, num_clusters=number_of_zones, plot=True)
    st.success("Work zones assigned.")
    unique_zones = sorted(set(e['work_zone'] for e in all_elements_dict.values() if e['work_zone'] != 'Unassigned'))
    st.write(f"**Number of unique work zones found:** {len(unique_zones)}")
    st.subheader("Work Zones Plot")
    if work_zones_fig:
        st.plotly_chart(work_zones_fig, use_container_width=True)

    # Optionally show a table of elements
    with st.expander("Elements Data Table", expanded=True):
        df_elements = pd.DataFrame(list(all_elements_dict.values()))
        st.dataframe(df_elements)

    # Load the WBS file
    st.markdown("---")
    st.markdown("### Load Work Breakdown Structure (WBS)")
    wbs_dir = st.text_input("Enter the directory to load WBS files", "./wbs")
    wbs_df = load_wbs(wbs_dir)

    if wbs_df is None:
        st.error("Failed to load WBS file.")
        st.stop()

    st.success("WBS loaded successfully.")
    with st.expander("WBS Data Overview before Filtering", expanded=False):
        st.dataframe(wbs_df)
        st.markdown("---")
    with st.expander("Unique Names in WBS"):
        unique_names_in_wbs = sorted(set(wbs_df['Source Qty'].dropna().astype(str).unique()))
        st.table(pd.DataFrame(unique_names_in_wbs, columns=["Unique Names in WBS"]))

    # Replace 'Parent.Quantity' with None in 'Source Qty' column
    wbs_df['Source Qty'] = wbs_df['Source Qty'].replace('Parent.Quantity', None)
    # Fill down in 'Source Qty' column
    wbs_df['Source Qty'] = wbs_df['Source Qty'].ffill()
    # 'Input Unit' is 'Units' split by '/' and take the last part
    wbs_df['Input Unit'] = wbs_df['Units'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else None)
    # Keep only relevant columns
    wbs_df = wbs_df[['Source Qty', 'Unit', 'Input Unit', 'Consumption']]
    # Filter out columns that are not 'HR' in Unit
    wbs_df = wbs_df[wbs_df['Unit'] == 'HR']

    wbs_df['Source Qty'] = wbs_df['Source Qty'].astype(str).str.split('.').str[:-1].str.join('.')  # Split by '.' and join all but last part

    wbs_df = wbs_df[wbs_df['Source Qty'].apply(lambda x: any(name in str(x) for name in unique_names))]
    wbs_df = wbs_df[wbs_df['Input Unit'] != 'TON']

    st.markdown("### Filtered WBS Data")
    st.dataframe(wbs_df)

    st.markdown("---")
    st.markdown("### Calculate Total Work Hours for Each Element")

    # Calculate total work hours for each element
    # First combine the 'Consumption' for each unique pair of 'Source Qty' and 'Input Unit'
    wbs_df['Consumption'] = pd.to_numeric(wbs_df['Consumption'], errors='coerce')
    total_work_hours = wbs_df.groupby(['Source Qty', 'Input Unit'])['Consumption'].sum().reset_index()
    
    st.markdown("### Total Work Hours for Each Pair of Source Qty and Input Unit")
    st.dataframe(total_work_hours)

    # Calculate total work hours for each element
    # Assume 'Consumption' is in hours
    for _, element in df_elements.iterrows():
        mask_length = (total_work_hours['Source Qty'] == element['name']) & (total_work_hours['Input Unit'] == 'LF')
        length_consumption = total_work_hours.loc[mask_length, 'Consumption'].sum()
        mask_area = (total_work_hours['Source Qty'] == element['name']) & (total_work_hours['Input Unit'] == 'SF')
        area_consumption = total_work_hours.loc[mask_area, 'Consumption'].sum()
        mask_volume = (total_work_hours['Source Qty'] == element['name']) & (total_work_hours['Input Unit'] == 'CY')
        volume_consumption = total_work_hours.loc[mask_volume, 'Consumption'].sum()
        # mask_weight = (total_work_hours['Source Qty'] == element['name']) & (total_work_hours['Input Unit'] == 'TON')
        # weight_consumption = total_work_hours.loc[mask_weight, 'Consumption'].sum()
        quanity_mask = (total_work_hours['Source Qty'] == element['name']) & (total_work_hours['Input Unit'] == 'EA')
        quantity_consumption = total_work_hours.loc[quanity_mask, 'Consumption'].sum()

        # Update the element with the calculated work hours
        length_hours = length_consumption * element['length']
        area_hours = area_consumption * element['area']
        volume_hours = volume_consumption * element['volume']
        quantity_hours = quantity_consumption * 1 # Assuming quantity is one per element
        # weight_hours = weight_consumption * 0.6  # Assuming weight is converted to hours with a factor of 0.6

        # total_hours = length_hours + area_hours + volume_hours + weight_hours
        total_hours = length_hours + area_hours + volume_hours + quantity_hours
        # Store the result in the DataFrame
        df_elements.at[element.name, 'total_work_hours'] = total_hours

    st.markdown("### Total Work Hours for Each Element")
    st.dataframe(df_elements[['name', 'total_work_hours']])

    st.markdown("---")
    with st.expander("Full Elements Data with Work Hours"):
        st.dataframe(df_elements)

    st.markdown("---")
    st.markdown("### Build the Network Graph")

    from build_graph import build_wbs_graph, shortest_path, build_gds_graph

    # graph_fig, edges = build_wbs_graph(df_elements=df_elements[df_elements['work_zone']==1])

    # st.plotly_chart(graph_fig, use_container_width=True)
    st.success("Network graph built successfully.")

    # st.dataframe(edges)

    build_gds_graph()
    short_path = shortest_path('242065', '235678', 'inv_time')
    st.markdown(short_path)

    # import gravis as gv
    # renderer = gv.three(
    #             G,
    #             use_node_size_normalization=True, 
    #             node_size_normalization_max=30,
    #             use_edge_size_normalization=True,
    #             edge_size_data_source='weight', 
    #             edge_curvature=0.3,
    #             node_hover_neighborhood=True,
    #             show_edge_label=True,
    #             edge_label_data_source='weight',
    #             node_label_size_factor=0.5,
    #             edge_size_factor=0.5,
    #             edge_label_size_factor=0.5,
    #             node_size_data_source='depth',
    #             layout_algorithm_active=True,
    #             # use_links_force=True,
    #             # links_force_distance=200,
    #             use_many_body_force=True,
    #             many_body_force_strength=-300,
    #             zoom_factor=1.5,
    #             graph_height=550,
    #         )
    # st.components.v1.html(renderer.to_html(), height=550)