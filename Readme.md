# Multi Robot Path Planning

This repository provides two planners for **multi-robot coordination** on grid-based maps:

* **Fragment Planner** – spatial coordination based on fragment-level path reservation.
* **Space–Time Planner** – an extended, time-aware version that accounts for movement delays and scheduling conflicts.

Both planners visualize multi-agent motion and can save animations or plots for analysis.

---

## 📁 Repository Structure

```
data/
 ├── map/                   # Default smaller environment
 └── map_bigger/            # Larger map for extended tests
figs/                       # Saved plots and figures
output/                     # Saved animations (if enabled)
src/
 ├── main.py                # Run Fragment Planner
 ├── main_st.py             # Run Space–Time Planner
 ├── planner/               # Core fragment planner modules
 ├── space_time_planner/    # Core space–time planner modules
 ├── metrics/               # Performance metrics
 └── utils/                 # Helper functions
diagram/                    # Architecture diagram
requirements.txt
```

---

## ⚙️ Setup

**Requirements:**

* Python 3.8+
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

---

## ▶️ Usage

### Run the Fragment Planner

```bash
python src/main.py
```

### Run the Space–Time Planner

```bash
python src/main_st.py
```

**Optional flags:**

| Flag        | Description                 | Example                                |
| ----------- | --------------------------- | -------------------------------------- |
| `--map`     | Choose a map file           | `--map data/map_bigger.yaml` |
| `--save`    | Save animation to `/output` | `--save True`                          |
| `--animate` | Show or hide live animation | `--animate True`                      |

Example:

```bash
python src/main_st.py --map data/map_bigger.yaml --save True --animate False
```

---

## 📊 Outputs

* **Plots** → saved in `figs/`
* **Animations** → saved in `output/`
* **Terminal logs** → show step-by-step execution and completion info

---

## 🧩 Notes

* The **Fragment Planner** handles spatial path coordination using local reservation and release rules.
* The **Space–Time Planner** integrates timing awareness for smoother, delay-minimized navigation.
* The structure is modular and easy to extend with new planners or maps.

---

## 🌻 Acknowledgment

Developed at the **University of Lincoln**
Supervised by **Dr. Gautham Das**
