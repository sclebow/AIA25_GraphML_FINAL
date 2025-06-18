import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd

from main import load_ifc_file, create_all_elements_dict, assign_levels, assign_work_zones
from dependency_utils import get_wbs_from_directory, load_wbs

st.set_page_config(page_title="IFC Level & Work Zone Visualizer", layout="wide")
st.title("IFC Level & Work Zone Visualizer")

# File selection from ./ifc directory
ifc_dir = "./ifc"
if not os.path.exists(ifc_dir):
    st.error(f"IFC directory '{ifc_dir}' not found.")
    st.stop()

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
        df = pd.DataFrame(list(all_elements_dict.values()))
        st.dataframe(df)

    # Load the WBS file
    st.markdown("---")
    st.markdown("### Load Work Breakdown Structure (WBS)")
    wbs_dir = st.text_input("Enter the directory to load WBS files", "./wbs")
    wbs_df = load_wbs(wbs_dir)

    if wbs_df is None:
        st.error("Failed to load WBS file.")
        st.stop()

    st.success("WBS loaded successfully.")

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

    st.dataframe(wbs_df)

    # Find matching elements between WBS and Element Names
    source_qty_list = wbs_df['Source Qty'].tolist()
    print('element columns is {}'.format(df_elements.columns))
    # df_elements['hours'] = wbs_df.loc[wbs_df['Source Qty'].isin(df_elements['name']), 'Consumption'].values
