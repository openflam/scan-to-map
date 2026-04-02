# Benchmark Collection Guide

This guide explains how to use the Benchmark Collection UI to create challenging, high-quality queries and evaluate expected answers within 3D environments.

## Accessing the Interface
Navigate to the `/benchmark_collection` route to access the Benchmark UI. If a dataset is not specified in the URL, you will be prompted to select a dataset via the dataset picker page.

## Exploring the 3D Environment
Before writing benchmarks, you need to understand the scene and locate relevant objects (components). There are several interaction methods provided:

### 1. Download Annotations
Click the **"Download Annotations"** button to load all annotated bounding boxes for the current dataset. This fetches all the available interactive objects and surfaces them in the environment. 
* Once downloaded, checking **"Show Auto Tags"** will display the object tags hovering over each bounding box.

### 2. Search for Components
Use the search bar at the top of the interface. The Benchmark UI uses **BM25** searching to help you look up objects by text or keywords. The results will populate in the left sidebar and focus inside the 3D viewer.

### 3. Identify and Click Components
You can click on any bounding box within the 3D scene (or within the search results sidebar). Clicking a component will:
* Point the camera directly at the object.
* Open a Component Details panel displaying a cropped image, the object's associated text caption, and its physical dimensions.
* Reveal the **Component ID**, which is strictly required for writing expected benchmark answers.

## Writing Benchmarks
The left sidebar contains the Benchmark Input panel, divided into two distinct sections:

### Question Area
Write the natural language query as naturally as a user would ask it. This can span from simple object inquiries ("Where is the microwave?") to complex or spatial tasks ("I need to wash my hands and dry them, where should I go?").

### Expected Answer Area
Write the ideal reasoning and answer to the question. Because the system strictly identifies objects, you must appropriately tag the item instances using their *Component ID* directly in your response.

**Formatting Component Tags:**
When referencing an object, encapsulate the noun inside a component tag using its ID. The syntax is `<component_ID>text</component_ID>`.
> **Example:** "You can use the `<component_23>coffee machine</component_23>` located next to the `<component_45>fridge</component_45>`."

### Testing with Preview
Directly above the Expected Answer box is a **"Preview"** toggle button. When clicked:
* Your raw tags will be parsed into clickable links.
* You can click these links to verify that they correspond to the physical components you intended to reference. Clicking the links will navigate the 3D camera to the requested ID. 
* If a Component ID does not exist, the preview will highlight it with an underline and trigger a console warning.

### Saving
Once your Question and Expected Answer are properly formatted, click **Save** at the bottom of the form to submit your benchmark.

## Ignored Controls
While navigating the UI, you may notice toggles for **"Show Occupancy Grid"** and **"Directions"**. You can safely ignore these completely, as they are exclusively related to the model's navigation and pathfinding demonstrations and are not required or utilized when collecting component-based benchmarks.
