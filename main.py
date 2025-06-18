# This is a python script that uses a user provided IFC file with task attributes
# It uses a lookup table to determine which tasks are dependent on which other tasks
# It uses the dependency information and the building levels and the work zones to create a directed acyclic graph (DAG)

import ifcopenshell
import ifcopenshell.util.element
from pprint import pprint

def load_ifc_file(file_path):
    """
    Load an IFC file and return the IFC object.
    
    :param file_path: Path to the IFC file.
    :return: Loaded IFC object.
    """
    try:
        ifc_file = ifcopenshell.open(file_path)
        return ifc_file
    except Exception as e:
        print(f"Error loading IFC file: {e}")
        return None
    
def load_latest_ifc_file(directory):
    """
    Load the latest IFC file from a given directory.
    
    :param directory: Directory containing IFC files.
    :return: Loaded IFC object or None if no files found.
    """
    import os
    ifc_files = [f for f in os.listdir(directory) if f.endswith('.ifc')]
    if not ifc_files:
        print("No IFC files found in the directory.")
        return None
    
    latest_file = max(ifc_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return load_ifc_file(os.path.join(directory, latest_file))

def create_all_elements_dict(ifc_file):
    """
    Create a dictionary of all elements in the IFC file.
    
    :param ifc_file: Loaded IFC object.
    :return: Dictionary of all elements.
    """
    def create_element_dict(element):
        """
        Create a dictionary representation of an IFC element.
        
        :param element: IFC element to convert.
        :return: Dictionary representation of the element.
        """
        element_dict = {
            'id': element.id(),
            'type': element.is_a(),
            'name': getattr(element, 'Name', None),
            'location': {
                'x': element.ObjectPlacement.RelativePlacement.Location.Coordinates[0],
                'y': element.ObjectPlacement.RelativePlacement.Location.Coordinates[1],
                'z': element.ObjectPlacement.RelativePlacement.Location.Coordinates[2]
            }
        }
        return element_dict

    elements = ifc_file.by_type("IfcElement")
    all_elements_dict = {}
    for element in elements:
        element_dict = create_element_dict(element)
        all_elements_dict[element.id()] = element_dict
    return all_elements_dict

def assign_levels(all_elements_dict, ifc_file, plot=False):
    """
    Assign levels to elements in the dictionary based on their placement using clustering.  
    Using a clustering algorithm to group elements by their Z-coordinate without defining a specific threshold or level count.

    :param all_elements_dict: Dictionary of all elements.
    :param ifc_file: Loaded IFC object.
    :param plot: Whether to plot the results using plotly.
    :return: Dictionary with levels assigned to elements.
    """
    
    from sklearn.cluster import DBSCAN
    import numpy as np
    if plot:
        import plotly.graph_objects as go

    # Extract Z-coordinates from element locations
    z_coordinates = np.array([element['location']['z'] for element in all_elements_dict.values()]).reshape(-1, 1)
    # Use DBSCAN to cluster elements based on their Z-coordinates
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(z_coordinates)
    labels = clustering.labels_

    # Reorder cluster labels so that level 1 is the lowest Z cluster
    import collections
    label_to_zs = collections.defaultdict(list)
    for label, element in zip(labels, all_elements_dict.values()):
        if label != -1:
            label_to_zs[label].append(element['location']['z'])
    # Compute mean Z for each cluster label
    label_mean_z = {label: np.mean(zs) for label, zs in label_to_zs.items()}
    # Sort labels by mean Z (ascending)
    sorted_labels = sorted(label_mean_z, key=lambda l: label_mean_z[l])
    # Map old labels to new levels (starting from 1)
    label_to_level = {label: i+1 for i, label in enumerate(sorted_labels)}

    # Assign new levels
    for element, label in zip(all_elements_dict.values(), labels):
        if label == -1:
            element['level'] = -1
        else:
            element['level'] = label_to_level[label]

    if plot:
        # Prepare data for plotting
        xs = [element['location']['x'] for element in all_elements_dict.values()]
        ys = [element['location']['y'] for element in all_elements_dict.values()]
        zs = [element['location']['z'] for element in all_elements_dict.values()]
        levels = [element['level'] for element in all_elements_dict.values()]
        names = [element['name'] for element in all_elements_dict.values()]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    size=6,
                    color=levels,
                    colorscale='Viridis',
                    colorbar=dict(title='Level'),
                    opacity=0.8
                ),
                text=names,
                hoverinfo='text+x+y+z'
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            title='IFC Elements by Level (DBSCAN Clustering)',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        # Show the plot interactively
        fig.show()
        # Save the plot as an HTML file
        fig.write_html("./plots/levels_plot.html")
        print("Plot saved as plot.html. Open this file in your browser to view the plot.")

    return all_elements_dict, fig if plot else None

def assign_work_zones(all_elements_dict, ifc_file, plot=False):
    """
    Assign work zones to elements in the dictionary based on their placement.
    Using a clustering algorithm to group elements by their X and Y coordinates, defining a specific threshold for work zones, but not a specific number of zones.

    :param all_elements_dict: Dictionary of all elements.
    :param ifc_file: Loaded IFC object.
    :param plot: Whether to plot the results using plotly.
    :return: Dictionary with work zones assigned to elements.
    """
    from sklearn.cluster import DBSCAN
    import numpy as np
    if plot:
        import plotly.graph_objects as go

    # Extract X and Y coordinates from element locations
    coordinates = np.array([[element['location']['x'], element['location']['y']] for element in all_elements_dict.values()])
    
    threshold = 6.0  # Define a threshold for clustering (can be adjusted)
    # Use DBSCAN to cluster elements based on their X and Y coordinates
    clustering = DBSCAN(eps=threshold, min_samples=2).fit(coordinates)
    labels = clustering.labels_

    # Assign work zones based on cluster labels
    for element, label in zip(all_elements_dict.values(), labels):
        if label == -1:
            element['work_zone'] = 'Unassigned'
        else:
            element['work_zone'] = f'Zone {label + 1}'

    # Reorder the work zone labels to ensure consistent naming
    # The order is determined by sum of X and Y coordinates for each zone
    zone_to_coords = {}
    for element, label in zip(all_elements_dict.values(), labels):
        if label != -1:
            zone_key = f'Zone {label + 1}'
            if zone_key not in zone_to_coords:
                zone_to_coords[zone_key] = []
            zone_to_coords[zone_key].append((element['location']['x'], element['location']['y']))

    # Compute mean coordinates for each zone
    zone_mean_coords = {zone: np.mean(coords, axis=0) for zone, coords in zone_to_coords.items()}

    # Sort zones by their mean coordinates (ascending)
    sorted_zones = sorted(zone_mean_coords, key=lambda z: (zone_mean_coords[z][0], zone_mean_coords[z][1]))

    # Create a mapping from old zone names to new ordered names
    zone_to_new_name = {zone: f'Zone {i + 1}' for i, zone in enumerate(sorted_zones)}

    # Assign new zone names
    for element in all_elements_dict.values():
        if element['work_zone'] != 'Unassigned':
            element['work_zone'] = zone_to_new_name[element['work_zone']]
    
    if plot:
        # Prepare data for plotting
        xs = [element['location']['x'] for element in all_elements_dict.values()]
        ys = [element['location']['y'] for element in all_elements_dict.values()]
        zs = [element['location']['z'] for element in all_elements_dict.values()]
        work_zones = [element['work_zone'] for element in all_elements_dict.values()]
        names = [element['name'] for element in all_elements_dict.values()]
        # Map work zone names to integers for coloring
        unique_zones = sorted(set(work_zones))
        zone_to_int = {zone: i for i, zone in enumerate(unique_zones)}
        work_zone_ints = [zone_to_int[zone] for zone in work_zones]
        hover_texts = [f"{name} ({zone})" for name, zone in zip(names, work_zones)]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    size=6,
                    color=work_zone_ints,
                    colorscale='Viridis',
                    colorbar=dict(title='Work Zone'),
                    opacity=0.8
                ),
                text=hover_texts,
                hoverinfo='text+x+y+z'
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            title='IFC Elements by Work Zone (DBSCAN Clustering)',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        # Show the plot interactively
        fig.show()
        # Save the plot as an HTML file
        fig.write_html("./plots/work_zones_plot.html")
        print("Plot saved as work_zones_plot.html. Open this file in your browser to view the plot.")

    if plot:
        return all_elements_dict, fig
    
    else:
        return all_elements_dict, None

def main():
    """
    Main function to execute the script.
    """
    ifc_directory = "./ifc" # Change this to your IFC directory
    ifc_file = load_latest_ifc_file(ifc_directory)
    print(f"Loaded IFC file: {ifc_file}")
    if not ifc_file:
        print("Failed to load IFC file.")
        return
    
    all_elements_dict = create_all_elements_dict(ifc_file)
    print(f"Created dictionary of all elements")
    all_elements_dict, fig = assign_levels(all_elements_dict, ifc_file, plot=True)
    print(f"Assigned levels to elements")
    print(f"Unique levels found: {set(element['level'] for element in all_elements_dict.values())}")
    all_elements_dict, fig = assign_work_zones(all_elements_dict, ifc_file, plot=True)
    print(f"Assigned work zones to elements")
    print(f"Unique work zones found: {set(element['work_zone'] for element in all_elements_dict.values())}")
    # pprint(all_elements_dict)

if __name__ == "__main__":
    main()