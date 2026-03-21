# 3D-Path-Planning-in-PGT
A procedural simulation framework for comparative analysis of 3D path planning algorithms, focusing on autonomous navigation in complex unknown terrains.

## Overview
This project presents a **Python-based simulation framework** for the comparative analysis of 3D path planning algorithms for autonomous mobile robots in unknown environments.

The system uses **procedural terrain generation** to evaluate algorithm performance across diverse and realistic scenarios, enabling efficient testing without physical hardware.

---

## Objectives
- Evaluate path planning in unknown 3D terrains
- Compare search-based and sampling-based algorithms
- Analyze trade-offs between speed, optimality, and exploration

---

## Algorithms Implemented
- **D* 3D** – Dynamic replanning with heuristic search  
- **RRT 3D** – Probabilistic exploration of unknown space  
- **RRT* 3D** – Optimized planning with asymptotic optimality  

---

## System Architecture
- Procedural terrain generation (Perlin Noise)
- Configurable environment (obstacles, start/goal)
- Simulation + visualization pipeline
- Performance evaluation metrics

---

## 🖼️ Results

### D* 3D

---

## 📊 Key Findings
- **D* 3D** → Fastest planning (~1.36s) and shortest path  
- **RRT 3D** → Strong exploration in unknown environments  
- **RRT* 3D** → Best path optimality but higher computation time (~6.81s) :contentReference[oaicite:0]{index=0}
