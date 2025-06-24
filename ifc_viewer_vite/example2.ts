/* MD
### 👓 Making things invisible
---

In this tutorial, you'll learn how control the visibility of the items of a BIM model.

:::tip Why make things invisible?

Many times, we just want to look at a specific part of a BIM model, without seeing the rest of it. BIM models are complex, and finding what we are looking for is not always easy. Luckily, the components library has tools to make it easier!

:::

In this tutorial, we will import:

- `@thatopen/ui` to add some simple and cool UI menus.
- `web-ifc` to get some IFC items.
- `@thatopen/components` to set up the barebone of our app.
- `Stats.js` (optional) to measure the performance of our app.
*/

import Stats from "stats.js";
import * as BUI from "@thatopen/ui";
import * as WEBIFC from "web-ifc";
import * as OBC from "@thatopen/components";
import * as THREE from "three";

const urlParams = new URLSearchParams(window.location.search);
const ifcUrl = urlParams.get('ifcUrl') || '';

/* MD
  ### 🌎 Setting up a simple scene
  ---

  We will start by creating a simple scene with a camera and a renderer. If you don't know how to set up a scene, you can check the Worlds tutorial.
*/

const container = document.getElementById("container")!;

const components = new OBC.Components();

const worlds = components.get(OBC.Worlds);

const world = worlds.create<
  OBC.SimpleScene,
  OBC.SimpleCamera,
  OBC.SimpleRenderer
>();

world.scene = new OBC.SimpleScene(components);
world.renderer = new OBC.SimpleRenderer(components, container);
world.camera = new OBC.SimpleCamera(components);

components.init();

world.camera.controls.setLookAt(12, 6, 8, 0, 0, -10);

world.scene.setup();

const grids = components.get(OBC.Grids);
grids.create(world);

/* MD

  We'll make the background of the scene transparent so that it looks good in our docs page, but you don't have to do that in your app!

*/

world.scene.three.background = null;

/* MD
  ### 🔎 Custom filters for your BIM models
  ---
  
  First, let's start by creating a `FragmentManager` instance and
  loading a simple fragment. If you haven't checked out the tutorial
  for that component yet, do it before going forward!
*/
const fragments = components.get(OBC.FragmentsManager);
const fragmentIfcLoader = components.get(OBC.IfcLoader);
await fragmentIfcLoader.setup()

let model: any; // <-- Add this line at the top level

async function loadIfc() {
  const file = await fetch(
    ifcUrl,
  );
  const data = await file.arrayBuffer();
  const buffer = new Uint8Array(data);
  model = await fragmentIfcLoader.load(buffer);
  model.name = "example";
  world.scene.three.add(model);
}

await loadIfc();

// Move the grid to the lowest point in the model
if (model) {
  // Compute bounding box
  const box = new THREE.Box3().setFromObject(model);
  const minY = box.min.y;
  // Shift model up so its lowest point is at Y=0.01
  if (minY < 0.01) {
    model.position.y += (0.01 - minY);
  } else {
    model.position.y -= (minY - 0.01);
  }

  // --- Camera framing logic ---
  // Compute bounding sphere to get center and radius
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const center = sphere.center;
  const radius = sphere.radius;

  // Calculate camera position: place it back along Z and up along Y
  // so the model is nicely visible (adjust factor as needed)
  const cameraDistance = radius * 2.2; // 2.2 gives some margin
  const cameraPos = new THREE.Vector3(
    center.x + cameraDistance * 0.7,
    center.y + cameraDistance * 0.5,
    center.z + cameraDistance
  );

  // Set camera position and aim
  world.camera.controls.setLookAt(
    cameraPos.x, cameraPos.y, cameraPos.z,
    center.x, center.y, center.z
  );
}

fragments.onFragmentsLoaded.add((model) => {
  console.log(model);
});


// const fragments = components.get(OBC.FragmentsManager);
// const file = await fetch(
//   "https://thatopen.github.io/engine_components/resources/small.frag",
// );
// const data = await file.arrayBuffer();
// const buffer = new Uint8Array(data);
// const model = fragments.load(buffer);
// world.scene.three.add(model);

// const properties = await fetch(
//   "https://thatopen.github.io/engine_components/resources/small.json",
// );
// model.setLocalProperties(await properties.json());

const indexer = components.get(OBC.IfcRelationsIndexer);
await indexer.process(model);

/* MD
  Now that we have our model, let's start the `FragmentHider`. You
  can use the `loadCached` method if you had used it before: it will
  automatically load all the filters you created in previous sessions,
  even after closing the browser and opening it again:
*/

const hider = components.get(OBC.Hider);

/* MD
  ### 📕📗📘 Setting up simple filters
  ---
  Next, we will classify data by category using the `Classifier`. This will allow us to create a simple filter for categories only.
*/

const classifier = components.get(OBC.Classifier);
classifier.byEntity(model);
/* MD
  ### ⏱️ Measuring the performance (optional)
  ---

  We'll use the [Stats.js](https://github.com/mrdoob/stats.js) to measure the performance of our app. We will add it to the top left corner of the viewport. This way, we'll make sure that the memory consumption and the FPS of our app are under control.
*/

const stats = new Stats();
stats.showPanel(2);
document.body.append(stats.dom);
stats.dom.style.left = "0px";
stats.dom.style.zIndex = "unset";
world.renderer.onBeforeUpdate.add(() => stats.begin());
world.renderer.onAfterUpdate.add(() => stats.end());

/* MD
  ### 🧩 Adding some UI
  ---

  We will use the `@thatopen/ui` library to add some simple and cool UI elements to our app. First, we need to call the `init` method of the `BUI.Manager` class to initialize the library:
*/

BUI.Manager.init();

/* MD
  Now, let's create a simple object for categories only:
*/

const classes: Record<string, any> = {};
const classNames = Object.keys(classifier.list.entities);
for (const name of classNames) {
  classes[name] = true;
}

// After classifier.byEntity(model);

// Define categoryNames and defaultColors at the top-level scope
const categoryNames = Object.keys(classes).filter(name => name !== 'IFCSPACE');
const numberOfCategories = categoryNames.length;
const defaultColors = categoryNames.map((_, i) => {
  const hue = i / numberOfCategories;
  return new THREE.Color().setHSL(hue, 0.5, 0.5);
});

// Assign default colors to each category after model is loaded
for (const name of classNames) {
  classes[name] = true;
}

// Assign default colors to each category after model is loaded
(async () => {
  // Generate evenly distributed colors for categories (except IFCSPACE)
  const categoryNames = Object.keys(classes).filter(name => name !== 'IFCSPACE');
  const numberOfCategories = categoryNames.length;
  const defaultColors = categoryNames.map((_, i) => {
    const hue = i / numberOfCategories;
    return new THREE.Color().setHSL(hue, 0.5, 0.5);
  });

  // Set default color for each category
  categoryNames.forEach((name, i) => {
    const found = classifier.find({ entities: [name] });
    for (const fragmentID in found) {
      const fragment = fragments.list.get(fragmentID);
      if (!fragment) continue;
      let meshes: any[] = [];
      if (Array.isArray(fragment.mesh)) {
        meshes = fragment.mesh;
      } else if (fragment.mesh) {
        meshes = [fragment.mesh];
      }
      const expressIDs = Array.from(found[fragmentID]);
      // Create a transparent material for this category
      const color = defaultColors[i];
      const material = new THREE.MeshStandardMaterial({
        color: color,
        transparent: true,
        opacity: 0.5
      });
      for (const mesh of meshes) {
        // For instanced meshes, set color per instance and assign material
        if (mesh && mesh.isInstancedMesh && mesh.instanceColor && fragment.itemToInstances) {
          mesh.material = material;
          for (const expressID of expressIDs) {
            const instances = fragment.itemToInstances.get(expressID);
            if (!instances) continue;
            for (const instanceIndex of instances) {
              mesh.instanceColor.setXYZ(instanceIndex, color.r, color.g, color.b);
            }
          }
          mesh.instanceColor.needsUpdate = true;
        } else if (mesh) {
          // For non-instanced meshes, assign the material directly
          mesh.material = material;
        }
      }
    }
  });

  // Set IFCSPACE to transparent gray
  if (classes['IFCSPACE']) {
    const found = classifier.find({ entities: ['IFCSPACE'] });
    for (const fragmentID in found) {
      const fragment = fragments.list.get(fragmentID);
      if (!fragment) continue;
      let meshes: any[] = [];
      if (Array.isArray(fragment.mesh)) {
        meshes = fragment.mesh;
      } else if (fragment.mesh) {
        meshes = [fragment.mesh];
      }
      const expressIDs = Array.from(found[fragmentID]);
      const material = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa,
        transparent: true,
        opacity: 0.1
      });
      for (const mesh of meshes) {
        if (mesh && mesh.isInstancedMesh && mesh.instanceColor && fragment.itemToInstances) {
          mesh.material = material;
          for (const expressID of expressIDs) {
            const instances = fragment.itemToInstances.get(expressID);
            if (!instances) continue;
            for (const instanceIndex of instances) {
              mesh.instanceColor.setXYZ(instanceIndex, 0.67, 0.67, 0.67); // gray
            }
          }
          mesh.instanceColor.needsUpdate = true;
        } else if (mesh) {
          mesh.material = material;
        }
      }
    }
  }
})();


/* MD
Now we will add some UI to control the visibility of items per category using simple checkboxes.
*/

// Create the panel and section separately
const panel = BUI.Component.create<BUI.Panel>(() => {
  return BUI.html`
    <bim-panel active label="Categories" class="options-menu"></bim-panel>
  `;
});

const categorySection = BUI.Component.create<BUI.PanelSection>(() => {
  return BUI.html`
    <bim-panel-section name="Categories"></bim-panel-section>
  `;
});

panel.append(categorySection);
document.body.append(panel);

// Apply default materials to the model based on categories
// IFCSPACE should always get the same default material
const defaultMaterial = new THREE.MeshStandardMaterial({
  color: 0xaaaaaa,
    transparent: true,
    opacity: 0.1
});

// The other categories will be set to evenly distributed colors, one for each category extracted from the model, except for IFCSPACE
const colors = Array.from({ length: numberOfCategories }, (_, i) => {
    const hue = i / numberOfCategories;
    return new THREE.Color().setHSL(hue, 0.5, 0.5);
});

// Set the default material for each category

// Store default color and opacity for each category
const categoryDefaults: Record<string, { color: THREE.Color; opacity: number }> = {};
const defaultOpacity = 0.5;
categoryNames.forEach((name, i) => {
  categoryDefaults[name] = {
    color: defaultColors[i],
    opacity: defaultOpacity,
  };
});
if (classes['IFCSPACE']) {
  categoryDefaults['IFCSPACE'] = {
    color: new THREE.Color(0xaaaaaa),
    opacity: 0.1,
  };
}

// Add a header row for the controls
const header = document.createElement('div');
header.style.display = 'flex';
header.style.alignItems = 'center';
header.style.fontWeight = 'bold';
header.style.marginBottom = '4px';

const labelCheckbox = document.createElement('span');
labelCheckbox.textContent = 'Category';
labelCheckbox.style.display = 'inline-block';
labelCheckbox.style.width = '120px';

const labelColor = document.createElement('span');
labelColor.textContent = 'Color';
labelColor.style.display = 'inline-block';
labelColor.style.width = '60px';
labelColor.style.textAlign = 'center';

const labelOpacity = document.createElement('span');
labelOpacity.textContent = 'Opacity';
labelOpacity.style.display = 'inline-block';
labelOpacity.style.width = '100px';
labelOpacity.style.textAlign = 'center';

header.appendChild(labelCheckbox);
header.appendChild(labelColor);
header.appendChild(labelOpacity);
categorySection.append(header);

for (const name in classes) {
  const defaultColor = categoryDefaults[name]?.color || new THREE.Color('#ffffff');
  const defaultOpacity = categoryDefaults[name]?.opacity ?? 0.5;
  const checkbox = BUI.Component.create<BUI.Checkbox>(() => {
    return BUI.html`
      <bim-checkbox checked label="${name}"
        @change="${({ target }: { target: BUI.Checkbox }) => {
          const found = classifier.find({ entities: [name] });
          hider.set(target.value, found);
        }}">
      </bim-checkbox>
    `;
  });
  checkbox.style.width = '120px';

  // Create a color input for material color
  const colorInput = document.createElement('input');
  colorInput.type = 'color';
  colorInput.style.marginLeft = '8px';
  colorInput.style.width = '60px';
  colorInput.value = '#' + defaultColor.getHexString();
  colorInput.title = `Change color for ${name}`;

  // Create a range input for opacity
  const opacityInput = document.createElement('input');
  opacityInput.type = 'range';
  opacityInput.min = '0';
  opacityInput.max = '1';
  opacityInput.step = '0.01';
  opacityInput.value = String(defaultOpacity);
  opacityInput.title = `Change opacity for ${name}`;
  opacityInput.style.marginLeft = '8px';
  opacityInput.style.width = '100px';

  // Handler to update color and opacity
  function updateMaterial() {
    const hex = colorInput.value;
    const opacity = parseFloat(opacityInput.value);
    const color = new THREE.Color(hex);
    const r = color.r, g = color.g, b = color.b;
    const found = classifier.find({ entities: [name] });
    for (const fragmentID in found) {
      const fragment = fragments.list.get(fragmentID);
      if (!fragment) continue;
      let meshes: any[] = [];
      if (Array.isArray(fragment.mesh)) {
        meshes = fragment.mesh;
      } else if (fragment.mesh) {
        meshes = [fragment.mesh];
      }
      const expressIDs = Array.from(found[fragmentID]);
      const material = new THREE.MeshStandardMaterial({
        color: color,
        transparent: true,
        opacity: opacity
      });
      for (const mesh of meshes) {
        if (mesh && mesh.isInstancedMesh && mesh.instanceColor && fragment.itemToInstances) {
          mesh.material = material;
          for (const expressID of expressIDs) {
            const instances = fragment.itemToInstances.get(expressID);
            if (!instances) continue;
            for (const instanceIndex of instances) {
              mesh.instanceColor.setXYZ(instanceIndex, r, g, b);
            }
          }
          mesh.instanceColor.needsUpdate = true;
        } else if (mesh) {
          mesh.material = material;
        }
      }
    }
  }

  colorInput.addEventListener('input', updateMaterial);
  opacityInput.addEventListener('input', updateMaterial);

  // Wrap checkbox, color input, and opacity slider in a div for layout
  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.alignItems = 'center';
  wrapper.style.marginBottom = '2px';
  wrapper.appendChild(checkbox);
  wrapper.appendChild(colorInput);
  wrapper.appendChild(opacityInput);

  categorySection.append(wrapper);
}

/* MD
  And we will make some logic that adds a button to the screen when the user is visiting our app from their phone, allowing to show or hide the menu. Otherwise, the menu would make the app unusable.
*/

const button = BUI.Component.create<BUI.PanelSection>(() => {
  return BUI.html`
      <bim-button class="phone-menu-toggler" icon="solar:settings-bold"
        @click="${() => {
          if (panel.classList.contains("options-menu-visible")) {
            panel.classList.remove("options-menu-visible");
          } else {
            panel.classList.add("options-menu-visible");
          }
        }}">
      </bim-button>
    `;
});

document.body.append(button);

/* MD
### 🎉 Wrap up
---

  That's it! You have created an app with an UI that allows the user to control the visibility of items in a BIM model by category only. Well done!
*/