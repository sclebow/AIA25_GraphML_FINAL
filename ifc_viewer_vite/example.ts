/* MD
### 🏢 Loading IFC files
---

IFC is complex, and sometimes we want to look for items using complex filters. For instance, imagine we want to target all items in a file that have a property called FireProtection. This is due to the indirection present in most IFC files. Luckily for you, we have a component to easily perform complex queries on any IFC: the IfcFinder. In this tutorial, you'll learn how to use it.

:::tip What does the finder do?

The finder is a powerful text scanner that can make complex queries in one or multiple IFC files. You can use regular expressions, operators, combine multiple filters, etc.

:::

In this tutorial, we will import:

- `@thatopen/ui` to add some simple and cool UI menus.
- `@thatopen/components` to set up the barebone of our app.
- `Stats.js` (optional) to measure the performance of our app.
*/

import * as BUI from "@thatopen/ui";
import Stats from "stats.js";
import * as OBC from "@thatopen/components";

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
  ### 🧳 Loading a BIM model
  ---

 We'll start by adding a BIM model to our scene. That model is already converted to fragments, so it will load much faster than if we loaded the IFC fileResponse.

  :::tip Fragments?

    If you are not familiar with fragments, check out the IfcLoader tutorial!

  :::
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

fragments.onFragmentsLoaded.add((model) => {
  console.log(model);
});

const hider = components.get(OBC.Hider);
