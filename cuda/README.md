# CUDA Understanding

<img width="1195" height="706" alt="image" src="https://github.com/user-attachments/assets/17498beb-136d-4c6c-9ecb-2fb0cbd91eb6" />

Here is the same write-up in a plain-text friendly format that will copy-paste perfectly into a README.

### The City of Blocks Analogy for CUDA Indexing

**The Concept**
Imagine a CUDA Grid as a City.

* The City (Grid) is made of Buildings (Blocks).
* Each Building (Block) contains Apartments (Threads).
* Our goal is to give every single apartment in the entire city a unique address (Global ID).

**Step 1: Identify the Building (block_id)**
First, we calculate a single number to identify which building we are in. We do this by counting all the buildings that come before ours in the grid.

* Logic:
* `blockIdx.x`: Move horizontally along the street (X).
* `blockIdx.y * gridDim.x`: Move down the rows of buildings (Y). To get here, we skip full rows, so we multiply our row number by the city's width.
* `blockIdx.z`: Move to deeper districts (Z). We skip entire 2D slices of the city.



```c
int block_id = blockIdx.x +
               (blockIdx.y * gridDim.x) +
               (blockIdx.z * gridDim.x * gridDim.y);

```

**Step 2: Count the Population Before Us (block_offset)**
Now that we know we are the Nth building, we calculate how many people live in all the buildings before ours.

* Logic: Multiply our Building Number by the total capacity of a single building.
* Analogy: (My Building ID) * (People per Building)

```c
int block_offset = block_id * (blockDim.x * blockDim.y * blockDim.z);

```

**Step 3: Identify the Apartment (thread_offset)**
Now we look inside our specific building to find our local apartment number. This follows the same X/Y/Z logic as Step 1, but constrained to the building's dimensions.

* Logic:
* `threadIdx.x`: Walk down the hall.
* `threadIdx.y * blockDim.x`: Go up the floors.
* `threadIdx.z`: Go deeper into the building.



```c
int thread_offset = threadIdx.x +
                    (threadIdx.y * blockDim.x) +
                    (threadIdx.z * blockDim.x * blockDim.y);

```

**Step 4: The Global Address (id)**
Finally, to get the unique ID, we simply add the two offsets.

* Logic: (Total people in previous buildings) + (My specific apartment number) = Global ID.

```c
int id = block_offset + thread_offset;

```

**Concrete Example**

* City Specs (Grid): 4 buildings wide (x=4).
* Building Specs (Block): 10 apartments per building (x=10).
* Our Location: We are in the building at index 2, inside that building at apartment index 5.

1. Block ID: We are at index 2.
2. Block Offset: 2 buildings * 10 apts/bldg = 20 (20 people live in the buildings before us).
3. Thread Offset: We are at index 5.
4. Global ID: 20 + 5 = 25.
