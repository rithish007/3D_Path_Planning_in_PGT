# 3D-Path-Planning-in-PGT
## Overview
This project presents a **Python-based simulation framework** for the comparative analysis of 3D path planning algorithms for autonomous mobile robots in unknown environments.

The system uses **procedural terrain generation** to evaluate algorithm performance across diverse and realistic scenarios, enabling efficient testing without physical hardware.

## Objectives
- Evaluate path planning in unknown 3D terrains
- Compare search-based and sampling-based algorithms
- Analyze trade-offs between speed, optimality, and exploration

## Algorithms Implemented
- **D* 3D** – Dynamic replanning with heuristic search  
- **RRT 3D** – Probabilistic exploration of unknown space  
- **RRT* 3D** – Optimized planning with asymptotic optimality  

## System Architecture
- Procedural terrain generation (Perlin Noise)
- Configurable environment (obstacles, start/goal)
- Simulation + visualization pipeline
- Performance evaluation metrics

## 🖼️ Results

### D* 3D
<img width="400" height="Auto" alt="DS3D_2DPLOT_PC" src="https://github.com/user-attachments/assets/f316a366-6fb0-4931-b8c1-2270ba1784c9" />

### RRT* 3D
<img width="400" height="Auto" alt="RRTS3DBIGENV" src="https://github.com/user-attachments/assets/243d2bb5-8a51-42ef-8280-8938b727ffdb" />

### RRT 3D
<img width="400" height="Auto" alt="RRT3D_2DPlot" src="https://github.com/user-attachments/assets/86556ff2-9a52-40a6-af11-2cdc62188d1a" />

### Perlin Noise Env Test
<img width="400" height="Auto" alt="3DPlanning" src="https://github.com/user-attachments/assets/7fd18150-5272-4348-b08e-44cfda3c4d4c" />


## 📊 Key Findings
- **D* 3D** → Fastest planning (~1.36s) and shortest path  
- **RRT 3D** → Strong exploration in unknown environments  
- **RRT* 3D** → Best path optimality but higher computation time (~6.81s) :contentReference[oaicite:0]{index=0}
