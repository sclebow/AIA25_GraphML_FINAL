# This is a python script that uses a user provided IFC file with task attributes
# It uses a lookup table to determine which tasks are dependent on which other tasks
# It uses the dependency information and the building levels and the work zones to create a directed acyclic graph (DAG)

import ifcopenshell
import ifcopenshell.util.element

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

def get_all_attributes(ifc_file):
    """
    Retrieve all attributes from the IFC file.
    
    :param ifc_file: Loaded IFC object.
    :return: Dictionary of attributes.
    """
    attributes = {}
    for entity in ifc_file.by_type('IfcPropertySet'):
        for prop in entity.HasProperties:
            attributes[prop.Name] = prop.NominalValue.wrappedValue if hasattr(prop.NominalValue, 'wrappedValue') else prop.NominalValue
    return attributes

def __main__():
    """
    Main function to execute the script.
    """
    ifc_directory = "./ifc" # Change this to your IFC directory
    ifc_file = load_latest_ifc_file(ifc_directory)

    if not ifc_file:
        print("Failed to load IFC file.")
        return
    
    attributes = get_all_attributes(ifc_file)
    
    print("Attributes found in the IFC file:")
    for key, value in attributes.items():
        print(f"{key}: {value}")