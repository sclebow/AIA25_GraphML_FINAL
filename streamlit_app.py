import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd
import json
import subprocess
import socket
import threading
import streamlit.components.v1 as components

from main import load_ifc_file, create_all_elements_dict, assign_levels, assign_work_zones, write_parameters_to_ifc, plot_critical_path
from dependency_utils import get_wbs_from_directory, load_wbs

# Print some newlines to the console for better readability
print("\n" * 5)
print("Starting Streamlit app...")
print()


def run_vite_server():
    def find_open_port(preferred_port, max_tries=20):
        """Find an open port, starting from preferred_port, up to max_tries."""
        port = preferred_port
        for _ in range(max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    port += 1
        raise RuntimeError(f"No open port found in range {preferred_port}-{port}")

    default_vite_port = 5173

    if "VITE_PORT" not in st.session_state:
        st.session_state["VITE_PORT"] = find_open_port(default_vite_port)

    def stream_subprocess_output(process):
        def stream(pipe):
            for line in iter(pipe.readline, b''):
                print("Flask Server:", line.decode(errors="replace").rstrip())
            pipe.close()
        threading.Thread(target=stream, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=stream, args=(process.stderr,), daemon=True).start()

    vite_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ifc_viewer_vite")
    # Use session_state to persist vite_process across reruns
    vite_process = st.session_state.get("vite_process", None)
    if vite_process is None or vite_process.poll() is not None:
        if not os.path.isdir(vite_dir):
            st.error(f"Vite directory not found: {vite_dir}. Please ensure 'ifc_viewer_vite' exists.")
            return "Vite directory missing."
        try:
            vite_process = subprocess.Popen(
                "npm run dev -- --host",
                cwd=vite_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True  # Use shell=True for Windows compatibility
            )
            st.session_state["vite_process"] = vite_process
            stream_subprocess_output(vite_process)
        except FileNotFoundError as e:
            st.error(f"Failed to start Vite server: {e}\nIs npm installed and on your PATH?")
            return f"Failed to start Vite: {e}"
        except Exception as e:
            st.error(f"Unexpected error starting Vite: {e}")
            return f"Failed to start Vite: {e}"

def show_ifcjs_viewer_vite(height=600, ifc_file_path=None):
    """Embed the local Vite IFC viewer in Streamlit via iframe, passing the latest IFC file URL."""
    VITE_PORT = st.session_state["VITE_PORT"]
    vite_url = f"http://localhost:{VITE_PORT}/?ifcUrl={ifc_file_path}"
    print(f"Vite URL: {vite_url}")
    components.html(f"""
        <iframe src='{vite_url}' width='100%' height='{height}' style='border:none;'></iframe>
    """, height=height)

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

latest_ifc_index = max(
    range(len(ifc_files)),
    key=lambda i: os.path.getmtime(os.path.join(ifc_dir, ifc_files[i]))
)
ifc_files.sort(key=lambda f: os.path.getmtime(os.path.join(ifc_dir, f)), reverse=False)
st.write(f"Latest IFC file: {ifc_files[latest_ifc_index]}")

selected_ifc = st.selectbox("Select IFC file", ifc_files, index=latest_ifc_index)
ifc_path = os.path.join(ifc_dir, selected_ifc)

print(f"Selected IFC file: {ifc_path}")

# Copy the selected IFC file to the Vite public directory and update the URL
vite_public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ifc_viewer_vite", "public")
os.makedirs(vite_public_dir, exist_ok=True)
public_ifc_path = os.path.join(vite_public_dir, selected_ifc)
if not os.path.exists(public_ifc_path) or os.path.getmtime(ifc_path) > os.path.getmtime(public_ifc_path):
    import shutil
    shutil.copy2(ifc_path, public_ifc_path)
    print(f"Copied {ifc_path} to {public_ifc_path}")

# The URL for the Vite app should be relative to the public directory
vite_ifc_url = f"/{selected_ifc}"

# Start the Vite server
run_vite_server()
show_ifcjs_viewer_vite(ifc_file_path=vite_ifc_url)

with st.spinner("Loading IFC file and processing..."):
    ifc_file = load_ifc_file(ifc_path)
    if ifc_file is None:
        st.error("Failed to load IFC file.")
        st.stop()

if ifc_file:
    all_elements_dict = create_all_elements_dict(ifc_file)

    # Filter elements by Name
    df_elements = pd.DataFrame.from_dict(all_elements_dict, orient='index')
    df_elements = df_elements[
        ~df_elements['name'].str.startswith(('LVL', '1000', 'Stair'), na=False)
    ]
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
    level_threshold = st.slider(
        "Level clustering threshold (Z, meters)",
        min_value=0.01, max_value=2.0, value=0.1, step=0.01
    )
    all_elements_dict, levels_fig = assign_levels(
        all_elements_dict, ifc_file, threshold=level_threshold, plot=True
    )
    st.success("Levels assigned.")
    unique_levels = sorted(set(e['level'] for e in all_elements_dict.values() if e['level'] != -1))
    st.write(f"**Number of unique levels found:** {len(unique_levels)}")
    st.subheader("Levels Plot")
    if levels_fig:
        st.plotly_chart(levels_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Assign Work Zones")
    number_of_zones = st.slider(
        "Number of work zones to assign", min_value=1, max_value=20, value=6, step=1
    )
    all_elements_dict, work_zones_fig = assign_work_zones(
        all_elements_dict, ifc_file, num_clusters=number_of_zones, plot=True
    )
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
    wbs_df['Source Qty'] = wbs_df['Source Qty'].ffill()
    wbs_df['Input Unit'] = wbs_df['Units'].apply(
        lambda x: x.split('/')[-1] if isinstance(x, str) else None
    )
    wbs_df = wbs_df[['Source Qty', 'Unit', 'Input Unit', 'Consumption']]
    wbs_df = wbs_df[wbs_df['Unit'] == 'HR']
    wbs_df['Source Qty'] = wbs_df['Source Qty'].astype(str).str.split('.').str[:-1].str.join('.')
    wbs_df = wbs_df[wbs_df['Source Qty'].apply(lambda x: any(name in str(x) for name in unique_names))]
    wbs_df = wbs_df[wbs_df['Input Unit'] != 'TON']

    # st.markdown("### Filtered WBS Data")
    # st.dataframe(wbs_df)

    # st.markdown("---")
    # st.markdown("### Calculate Total Work Hours for Each Element")

    # wbs_df['Consumption'] = pd.to_numeric(wbs_df['Consumption'], errors='coerce')
    # total_work_hours = wbs_df.groupby(['Source Qty', 'Input Unit'])['Consumption'].sum().reset_index()

    # st.markdown("### Total Work Hours for Each Pair of Source Qty and Input Unit")
    # st.dataframe(total_work_hours)

    # for _, element in df_elements.iterrows():
    #     mask_length = (
    #         (total_work_hours['Source Qty'] == element['name']) &
    #         (total_work_hours['Input Unit'] == 'LF')
    #     )
    #     length_consumption = total_work_hours.loc[mask_length, 'Consumption'].sum()
    #     mask_area = (
    #         (total_work_hours['Source Qty'] == element['name']) &
    #         (total_work_hours['Input Unit'] == 'SF')
    #     )
    #     area_consumption = total_work_hours.loc[mask_area, 'Consumption'].sum()
    #     mask_volume = (
    #         (total_work_hours['Source Qty'] == element['name']) &
    #         (total_work_hours['Input Unit'] == 'CY')
    #     )
    #     volume_consumption = total_work_hours.loc[mask_volume, 'Consumption'].sum()
    #     quanity_mask = (
    #         (total_work_hours['Source Qty'] == element['name']) &
    #         (total_work_hours['Input Unit'] == 'EA')
    #     )
    #     quantity_consumption = total_work_hours.loc[quanity_mask, 'Consumption'].sum()

    #     length_hours = length_consumption * element['length']
    #     area_hours = area_consumption * element['area']
    #     volume_hours = volume_consumption * element['volume']
    #     quantity_hours = quantity_consumption * 1
    #     total_hours = length_hours + area_hours + volume_hours + quantity_hours
    #     df_elements.at[element.name, 'total_work_hours'] = total_hours

    # st.markdown("### Total Work Hours for Each Element")
    # st.dataframe(df_elements[['name', 'total_work_hours']])

    # st.markdown("---")
    # with st.expander("Full Elements Data with Work Hours"):
    #     st.dataframe(df_elements)

    # st.markdown("---")
    # st.markdown("### Build the Network Graph")

    # from build_graph import build_wbs_graph, shortest_path, build_gds_graph, load_to_neo4j

    # graph_fig, edges, G = build_wbs_graph(df_elements=df_elements)
    # load_to_neo4j(G, reset=True)
    # st.success("Network graph built successfully.")

    # build_gds_graph()
    # short_path = shortest_path(242065, 235678)
    # costs = short_path[0]['costs']
    # longest_cost = sum(1 / cost for cost in costs if cost > 0)

    # with open('./gds_shortest_path.json', 'w') as f:
    #     json.dump(short_path[0], f)
    # st.json(short_path[0])

    # critical_nodes = short_path[0]['GlobalIdPath']
    # df_elements['critical'] = df_elements['id'].isin(critical_nodes)

    # st.markdown(f"### the critical path for work zone 1 is {longest_cost} hours")
    # st.dataframe(df_elements)

    # critical_fig = plot_critical_path(df_elements)

    # write_parameters_to_ifc(ifc_file, './updated_ifc', df_elements=df_elements)