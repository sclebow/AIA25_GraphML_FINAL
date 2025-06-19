# This is a python script that uses a user provided IFC file with task attributes
# It uses a lookup table to determine which tasks are dependent on which other tasks
# It uses the dependency information and the building levels and the work zones to create a directed acyclic graph (DAG)

import ifcopenshell
import ifcopenshell.util.element

import ifcopenshell.geom
import ifcopenshell.util.shape
from pprint import pprint
import os

import multiprocessing

from sklearn.cluster import DBSCAN, KMeans
import numpy as np

import plotly.graph_objects as go
from tqdm import tqdm

settings = ifcopenshell.geom.settings()


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
    unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
    element_dict = {}
    ifc_elements = ['IfcWall', 'IfcSlab', 'IfcBeam', 'IfcColumn', 'IfcFooting', 'IfcStair', 'IfcRamp']
    for element_type in ifc_elements:
        ifc_shapes = ifc_file.by_type(element_type)
        iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count(), include=ifc_shapes)

        for shape_geo in iterator:
            # print(dir(shape_geo))
            geometry = shape_geo.geometry
            (x,y,z) = ifcopenshell.util.shape.get_shape_bbox_centroid(shape_geo, geometry)

            element_dict[str(shape_geo.id)] = {
                'id': shape_geo.id,
                'type': shape_geo.type,
                'name': shape_geo.name,
                'location': {
                    'x': float(x),
                    'y': float(y),
                    'z': float(z)
                },
                # 'height': ifcopenshell.util.shape.get_z(geometry)*unit_scale,
                'length': ifcopenshell.util.shape.get_max_xyz(geometry)*unit_scale*3.28,  # Convert to feet
                'area': ifcopenshell.util.shape.get_max_side_area(geometry)*unit_scale**2 * 3.28**2,  # Convert to square feet
                'volume': ifcopenshell.util.shape.get_volume(geometry)*unit_scale**3 * 1.09**3,  # Convert to cubic yards
            }

    # def create_element_dict(element):
    #     """
    #     Create a dictionary representation of an IFC element.
        
    #     :param element: IFC element to convert.
    #     :return: Dictionary representation of the element.
    #     """
    #     # if element.is_a() not in ifc_elements:
    #     #     return None
    #     # else:
    #     shape = ifcopenshell.geom.create_shape(settings, element)
    #     occ_shape = shape.geometry

    #     element_dict = {
    #         'id': element.id(),
    #         'type': element.is_a(),
    #         'name': getattr(element, 'Name', None),
    #         'location': {
    #             'x': element.ObjectPlacement.RelativePlacement.Location.Coordinates[0],
    #             'y': element.ObjectPlacement.RelativePlacement.Location.Coordinates[1],
    #             'z': element.ObjectPlacement.RelativePlacement.Location.Coordinates[2]
    #         },
    #         'height': ifcopenshell.util.shape.get_z(occ_shape),
    #         'length': ifcopenshell.util.shape.get_max_xyz(occ_shape),
    #         'volume': ifcopenshell.util.shape.get_volume(occ_shape),
    #         'area': ifcopenshell.util.shape.get_max_side_area(occ_shape),

    #     }
    #     return element_dict

    # elements = ifc_file.by_type("IfcElement")
    # footings = ifc_file.by_type("IfcFooting")
    # elements.extend(footings)  # Include footings as well
    # all_elements_dict = {}
    # for element in elements:
    #     element_dict = create_element_dict(element)
    #     all_elements_dict[element.id()] = element_dict
    return element_dict

def assign_levels(all_elements_dict, ifc_file, threshold=0.1, plot=False):
    """
    Assign levels to elements in the dictionary based on their placement using clustering.  
    Using a clustering algorithm to group elements by their Z-coordinate without defining a specific threshold or level count.

    :param all_elements_dict: Dictionary of all elements.
    :param ifc_file: Loaded IFC object.
    :param plot: Whether to plot the results using plotly.
    :return: Dictionary with levels assigned to elements.
    """
    element_z_values = [element['location']['z'] for element in all_elements_dict.values()]
    # Extract Z-coordinates from element locations
    z_coordinates = np.array(element_z_values).reshape(-1, 1)
    print(len(z_coordinates), "Z-coordinates extracted from elements.")
    # Use DBSCAN to cluster elements based on their Z-coordinates
    clustering = DBSCAN(eps=threshold, min_samples=2).fit(z_coordinates)
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
        # fig.show()
        # Save the plot as an HTML file
        # fig.write_html("./plots/levels_plot.html")
        print("Plot saved as plot.html. Open this file in your browser to view the plot.")

    return all_elements_dict, fig if plot else None

def assign_work_zones(all_elements_dict, ifc_file, num_clusters=6, plot=False):
    """
    Assign work zones to elements in the dictionary based on their placement.
    Using a clustering algorithm to group elements by their X and Y coordinates without defining a specific threshold or zone count.

    :param all_elements_dict: Dictionary of all elements.
    :param ifc_file: Loaded IFC object.
    :param plot: Whether to plot the results using plotly.
    :return: Dictionary with work zones assigned to elements. 
    :return: Plotly figure if plot is True, otherwise None.
    """


    # Extract X and Y coordinates from element locations
    coordinates = np.array([[element['location']['x'], element['location']['y']] for element in all_elements_dict.values()])
    # Use KMeans to cluster elements based on their X and Y coordinates
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(coordinates)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Sort the cluster centers by their X and Y coordinates
    sorted_indices = np.lexsort((centers[:, 1], centers[:, 0]))
    sorted_labels = np.array(sorted_indices)
    # label_to_zone = {label: f"Zone {i+1}" for i, label in enumerate(sorted_labels)}
    label_to_zone = {label: i+1 for i, label in enumerate(sorted_labels)}

    # Assign work zones to elements
    for element, label in zip(all_elements_dict.values(), labels):
        if label == -1:
            element['work_zone'] = 'Unassigned'
        else:
            element['work_zone'] = label_to_zone[label]

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

        fig = go.Figure(data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    size=6,
                    color=work_zone_ints,
                    colorscale='portland',
                    colorbar=dict(title='Work Zone'),
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
            title='IFC Elements by Work Zone (KMeans Clustering)',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        # Show the plot interactively
        # fig.show()
        # Save the plot as an HTML file
        # fig.write_html("./plots/work_zones_plot.html")
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
    
    if not ifc_file:
        print("Failed to load IFC file.")
        return
    
    all_elements_dict = create_all_elements_dict(ifc_file)
    print(f"Created dictionary of all elements")
    all_elements_dict, fig = assign_levels(all_elements_dict, ifc_file, plot=True)
    print(f"Assigned levels to elements")
    all_elements_dict, fig = assign_work_zones(all_elements_dict, ifc_file, plot=True)
    print(f"Assigned work zones to elements")
    
if __name__ == "__main__":
    main()