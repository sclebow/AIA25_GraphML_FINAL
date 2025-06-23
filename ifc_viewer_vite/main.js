import * as OBC from "@thatopen/components";
import * as BUI from "@thatopen/ui";
import * as THREE from "three";

const urlParams = new URLSearchParams(window.location.search);
const ifcUrl = urlParams.get('ifcUrl') || '';

async function loadViewer() {
  const container = document.getElementById('container');
  if (!container) return;

  // Make container responsive
  container.style.width = '100vw';
  container.style.height = '100vh';
  container.style.position = 'fixed';
  container.style.top = '0';
  container.style.left = '0';
  container.style.margin = '0';
  container.style.padding = '0';

  // Initialize components
  const components = new OBC.Components();

  // Set up the world
  const worlds = components.get(OBC.Worlds);
  const world = worlds.create();

  world.scene = new OBC.SimpleScene(components);
  world.renderer = new OBC.SimpleRenderer(components, container);
  world.camera = new OBC.SimpleCamera(components);

  // Configure renderer
  const renderer = world.renderer.three;
  renderer.setClearColor(new THREE.Color(0.95, 0.96, 0.98));
  renderer.setSize(window.innerWidth, window.innerHeight);

  components.init();

  // Set up camera controls
  world.camera.controls.setLookAt(12, 6, 8, 0, 0, -10);

  // Add a directional light that follows the camera
  const cameraLight = new THREE.DirectionalLight(0xffffff, 1.0);
  cameraLight.position.copy(world.camera.three.position + new THREE.Vector3(0, 10, 0));
  world.scene.three.add(cameraLight);

  // Update the light position on each render
  renderer.setAnimationLoop(() => {
    cameraLight.position.copy(world.camera.three.position + new THREE.Vector3(0, 10, 0));
    renderer.render(world.scene.three, world.camera.three);
  });

  // Add grid and axes
  const grids = components.get(OBC.Grids);
  grids.create(world);

  // Set up lighting
  world.scene.setup();

  if (ifcUrl) {
    console.log('Loading IFC from URL:', ifcUrl);
    try {
      // Get IFC loader
      const ifcLoader = components.get(OBC.IfcLoader);
      await ifcLoader.setup();

      // Load IFC file
      const response = await fetch(ifcUrl);
      const data = await response.arrayBuffer();
      const buffer = new Uint8Array(data);
      const model = await ifcLoader.load(buffer);

      world.scene.three.add(model);

      // Apply materials and scaling
      model.traverse((child) => {
        if (child.isMesh) {
          // Use a simple material for better performance
          child.material = new THREE.MeshStandardMaterial({
            color: 0xeeeeee, 
            metalness: 0.3,
            roughness: 0.6
          });
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });

      // Scale model to fixed bounds
      const desiredSize = 10;
      const box = new THREE.Box3().setFromObject(model);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      if (maxDim > 0) {
        const scale = desiredSize / maxDim;
        model.scale.set(scale, scale, scale);
      }

      // Center model
      const boxScaled = new THREE.Box3().setFromObject(model);
      const center = new THREE.Vector3();
      boxScaled.getCenter(center);
      model.position.x -= center.x;
      model.position.y = -boxScaled.min.y;
      model.position.z -= center.z;

      // Fit camera to model
      const bbox = new THREE.Box3().setFromObject(model);
      const sphere = new THREE.Sphere();
      bbox.getBoundingSphere(sphere);
      
      const distance = sphere.radius * 2;
      world.camera.controls.setLookAt(
        sphere.center.x + distance,
        sphere.center.y + distance,
        sphere.center.z + distance,
        sphere.center.x,
        sphere.center.y,
        sphere.center.z
      );

      console.log('IFC model loaded successfully');

      // Highlight logic (only for IFC model meshes)
      let highlighted = null;
      let originalMaterial = null;

      function onPointerMove(event) {
        const rect = container.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1
        );
        const camera = world.camera.three;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(model.children, true);
        if (highlighted && originalMaterial) {
          highlighted.material = originalMaterial;
          highlighted = null;
          originalMaterial = null;
        }
        for (const intersect of intersects) {
          if (intersect.object.isMesh) {
            highlighted = intersect.object;
            originalMaterial = highlighted.material;
            highlighted.material = new THREE.MeshStandardMaterial({
              color: 0xffff00,
              metalness: 0.1,
              roughness: 0.6
            });
            break;
          }
        }
      }
      container.addEventListener('pointermove', onPointerMove);

    } catch (err) {
      console.error('Error loading IFC:', err);
      container.innerHTML += `<h4 style='color:red;'>Error loading IFC: ${err.message}</h4>`;
    }
  } else {
    container.innerHTML = '<h3>No IFC file URL provided.</h3>';
  }

  // Handle window resize
  window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    world.renderer.three.setSize(width, height); // Use .three here
    world.camera.updateAspect();
  });
}

loadViewer();
