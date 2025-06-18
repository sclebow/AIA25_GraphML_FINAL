import streamlit as st
import os
from main import *
import plotly.graph_objects as go

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

selected_ifc = st.selectbox("Select IFC file", ifc_files)
ifc_path = os.path.join(ifc_dir, selected_ifc)


if st.button("Process IFC File"):
    with st.spinner("Loading IFC file and processing..."):
        ifc_file = load_ifc_file(ifc_path)
        if ifc_file is None:
            st.error("Failed to load IFC file.")
            st.stop()
        all_elements_dict = create_all_elements_dict(ifc_file)
        # Add sliders for thresholds
        level_threshold = st.slider("Level clustering threshold (Z, meters)", min_value=0.01, max_value=2.0, value=0.1, step=0.01)
        all_elements_dict, levels_fig = assign_levels(all_elements_dict, ifc_file, threshold=level_threshold, plot=True)
        st.success("Levels assigned.")
        unique_levels = sorted(set(e['level'] for e in all_elements_dict.values() if e['level'] != -1))
        st.write(f"**Number of unique levels found:** {len(unique_levels)}")
        st.subheader("Levels Plot")
        if levels_fig:
            st.plotly_chart(levels_fig, use_container_width=True)
        zone_threshold = st.slider("Work zone clustering threshold (XY, meters)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
        all_elements_dict, work_zones_fig = assign_work_zones(all_elements_dict, ifc_file, threshold=zone_threshold, plot=True)
        st.success("Work zones assigned.")
        unique_zones = sorted(set(e['work_zone'] for e in all_elements_dict.values() if e['work_zone'] != 'Unassigned'))
        st.write(f"**Number of unique work zones found:** {len(unique_zones)}")
        st.subheader("Work Zones Plot")
        if work_zones_fig:
            st.plotly_chart(work_zones_fig, use_container_width=True)

        # Optionally show a table of elements
        if st.checkbox("Show elements table"):
            import pandas as pd
            df = pd.DataFrame(list(all_elements_dict.values()))
            st.dataframe(df)
