import sys
import os
import glob
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.engine.backend.space import Point2D, CoordinateBounds, Shooter, Target
from src.engine.backend.obstacles import BoxObstacle, CircleObstacle, PolygonObstacle
from src.engine.backend.game_state import GameState
from src.engine.backend.session_logger import TestSessionLogger
from src.engine.ui.renderer import ArenaRenderer


class GraphWarApp:
    """
    Interactive 2D Graphwar Simulation and Attempt-by-Attempt Playback GUI.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GraphWars-LLM Simulation & History Playback Engine")
        self.root.geometry("1180x780")
        self.root.configure(bg="#0F172A")

        # Application state
        self.current_preset = "pillar"
        self.game_state = GameState.create_preset(self.current_preset)

        # History playback state
        self.playback_mode = False
        self.loaded_session_data: Optional[Dict[str, Any]] = None
        self.current_attempt_idx = 0
        self.total_attempts = 0

        self._build_ui()
        self._setup_plot()
        self._bind_keyboard()
        self._refresh_outputs_list()
        self.refresh_display()

    def _build_ui(self):
        # Top Header & Controls
        top_bar = tk.Frame(self.root, bg="#1E293B", pady=8, padx=12)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            top_bar,
            text="⚔️ GraphWars-LLM",
            font=("Segoe UI", 12, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        title_lbl.pack(side=tk.LEFT, padx=(0, 15))

        # Mode Indicator
        self.mode_lbl = tk.Label(
            top_bar,
            text="[LIVE MODE]",
            font=("Segoe UI", 9, "bold"),
            fg="#22C55E",
            bg="#1E293B",
        )
        self.mode_lbl.pack(side=tk.LEFT, padx=(0, 15))

        # Live Map Preset Dropdown
        self.preset_lbl = tk.Label(
            top_bar, text="Map Preset:", font=("Segoe UI", 9), fg="#94A3B8", bg="#1E293B"
        )
        self.preset_lbl.pack(side=tk.LEFT, padx=(0, 4))

        self.preset_var = tk.StringVar(value="pillar")
        presets = ["pillar", "slalom", "bunker", "terrain_valley", "direct_shot"]
        self.preset_menu = ttk.Combobox(
            top_bar,
            textvariable=self.preset_var,
            values=presets,
            state="readonly",
            width=12,
        )
        self.preset_menu.pack(side=tk.LEFT, padx=(0, 10))
        self.preset_menu.bind("<<ComboboxSelected>>", self._on_preset_change)

        ttk.Separator(top_bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Playback JSON Selector
        hist_lbl = tk.Label(
            top_bar, text="📁 History Log:", font=("Segoe UI", 9), fg="#94A3B8", bg="#1E293B"
        )
        hist_lbl.pack(side=tk.LEFT, padx=(0, 4))

        self.output_files_var = tk.StringVar()
        self.output_files_menu = ttk.Combobox(
            top_bar,
            textvariable=self.output_files_var,
            state="readonly",
            width=26,
        )
        self.output_files_menu.pack(side=tk.LEFT, padx=(0, 6))
        self.output_files_menu.bind("<<ComboboxSelected>>", self._on_output_selected)

        browse_btn = tk.Button(
            top_bar,
            text="Open JSON...",
            command=self._browse_json_file,
            bg="#334155",
            fg="#F8FAFC",
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            padx=8,
            cursor="hand2",
        )
        browse_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Coordinate Hover Label
        self.coord_lbl = tk.Label(
            top_bar,
            text="Cursor: (0.00, 0.00)",
            font=("Consolas", 10),
            fg="#38BDF8",
            bg="#1E293B",
        )
        self.coord_lbl.pack(side=tk.RIGHT, padx=10)

        # Main Layout: Left = Plot + Playback Nav, Right = Sidebar
        main_frame = tk.Frame(self.root, bg="#0F172A")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Container
        left_container = tk.Frame(main_frame, bg="#0F172A")
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Plot Frame
        self.plot_frame = tk.Frame(left_container, bg="#0F172A")
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom Attempt Stepper Bar
        self.nav_bar = tk.Frame(left_container, bg="#1E293B", pady=6, padx=10)
        self.nav_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        self.prev_btn = tk.Button(
            self.nav_bar,
            text="◀ Previous (Left Arrow)",
            command=self._prev_attempt,
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#F8FAFC",
            relief=tk.FLAT,
            padx=12,
            cursor="hand2",
        )
        self.prev_btn.pack(side=tk.LEFT)

        self.attempt_step_lbl = tk.Label(
            self.nav_bar,
            text="Live Simulation Mode",
            font=("Segoe UI", 10, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        self.attempt_step_lbl.pack(side=tk.LEFT, expand=True)

        self.next_btn = tk.Button(
            self.nav_bar,
            text="Next (Right Arrow) ▶",
            command=self._next_attempt,
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#F8FAFC",
            relief=tk.FLAT,
            padx=12,
            cursor="hand2",
        )
        self.next_btn.pack(side=tk.RIGHT)

        # Right Sidebar Frame
        self.sidebar = tk.Frame(main_frame, bg="#1E293B", width=380, padx=14, pady=12)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.sidebar.pack_propagate(False)

        # Sidebar: Formula Input Section (Live Mode)
        self.input_section = tk.Frame(self.sidebar, bg="#1E293B")
        self.input_section.pack(fill=tk.X)

        input_title = tk.Label(
            self.input_section,
            text="Trajectory Input",
            font=("Segoe UI", 11, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        input_title.pack(anchor=tk.W, pady=(0, 4))

        input_row = tk.Frame(self.input_section, bg="#1E293B")
        input_row.pack(fill=tk.X, pady=(0, 8))

        y_prefix = tk.Label(
            input_row, text="y = ", font=("Consolas", 12, "bold"), fg="#38BDF8", bg="#1E293B"
        )
        y_prefix.pack(side=tk.LEFT)

        self.formula_var = tk.StringVar(value="-0.02*(x+20)*(x-20)")
        self.formula_entry = tk.Entry(
            input_row,
            textvariable=self.formula_var,
            font=("Consolas", 11),
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#38BDF8",
            relief=tk.FLAT,
        )
        self.formula_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(4, 0))
        self.formula_entry.bind("<Return>", lambda e: self._on_fire())

        btn_row = tk.Frame(self.input_section, bg="#1E293B")
        btn_row.pack(fill=tk.X, pady=(0, 10))

        fire_btn = tk.Button(
            btn_row,
            text="🚀 Fire Trajectory",
            command=self._on_fire,
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            relief=tk.FLAT,
            pady=5,
            cursor="hand2",
        )
        fire_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        clear_btn = tk.Button(
            btn_row,
            text="Clear",
            command=self._clear_live,
            font=("Segoe UI", 9),
            bg="#334155",
            fg="#F8FAFC",
            relief=tk.FLAT,
            pady=5,
            cursor="hand2",
        )
        clear_btn.pack(side=tk.RIGHT)

        # Divider
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill=tk.X, pady=8)

        # Sidebar: Attempt Analysis / Diagnostics Panel
        diag_title = tk.Label(
            self.sidebar,
            text="📊 Attempt Diagnostics & Reasoning",
            font=("Segoe UI", 11, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        diag_title.pack(anchor=tk.W, pady=(0, 6))

        # Structured Attempt Info Badges
        self.info_frame = tk.Frame(self.sidebar, bg="#0F172A", padx=8, pady=8)
        self.info_frame.pack(fill=tk.X, pady=(0, 8))

        self.strategy_lbl = tk.Label(
            self.info_frame,
            text="Strategy: None",
            font=("Segoe UI", 9, "bold"),
            fg="#A855F7",
            bg="#0F172A",
            anchor=tk.W,
        )
        self.strategy_lbl.pack(fill=tk.X)

        self.outcome_lbl = tk.Label(
            self.info_frame,
            text="Status: Ready",
            font=("Segoe UI", 9, "bold"),
            fg="#38BDF8",
            bg="#0F172A",
            anchor=tk.W,
        )
        self.outcome_lbl.pack(fill=tk.X, pady=(2, 0))

        # LLM Reasoning & State Text Area
        self.diag_text = tk.Text(
            self.sidebar,
            height=14,
            bg="#0F172A",
            fg="#E2E8F0",
            font=("Consolas", 9),
            relief=tk.FLAT,
            padx=8,
            pady=8,
            wrap=tk.WORD,
        )
        self.diag_text.pack(fill=tk.BOTH, expand=True)

    def _setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor("#0F172A")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.renderer = ArenaRenderer(self.ax, self.game_state.bounds)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _bind_keyboard(self):
        self.root.bind("<Left>", lambda e: self._prev_attempt())
        self.root.bind("<Right>", lambda e: self._next_attempt())

    def _refresh_outputs_list(self):
        items = []
        if os.path.exists("outputs"):
            for entry in sorted(os.listdir("outputs"), reverse=True):
                full_path = os.path.join("outputs", entry)
                if os.path.isdir(full_path):
                    hist_file = os.path.join(full_path, "history.json")
                    if os.path.exists(hist_file):
                        items.append(f"[Folder] {entry}")
                elif entry.endswith(".json") and entry != ".gitkeep":
                    items.append(entry)

        self.output_files_menu["values"] = items
        if items:
            self.output_files_menu.set(items[0])
            self._on_output_selected()

    def _on_output_selected(self, event=None):
        selected = self.output_files_var.get()
        if not selected:
            return
        
        if selected.startswith("[Folder] "):
            folder_name = selected.replace("[Folder] ", "").strip()
            filepath = os.path.join("outputs", folder_name, "history.json")
        else:
            filepath = os.path.join("outputs", selected)

        if os.path.exists(filepath):
            self._load_json_data(filepath)

    def _browse_json_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="outputs",
            title="Select Output History JSON",
            filetypes=[("JSON Files", "*.json")],
        )
        if filepath:
            self._load_json_data(filepath)

    def _load_json_data(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.loaded_session_data = data
            self.playback_mode = True
            self.mode_lbl.config(text="[HISTORY PLAYBACK]", fg="#A855F7")
            self.current_attempt_idx = 0
            self.total_attempts = len(data.get("history", []))

            # Reconstruct initial game state
            init_st = data.get("initial_state", {})
            bounds = CoordinateBounds(
                x_min=init_st["bounds"]["x_min"],
                x_max=init_st["bounds"]["x_max"],
                y_min=init_st["bounds"]["y_min"],
                y_max=init_st["bounds"]["y_max"],
            )
            shooter = Shooter(
                name=init_st["shooter"]["name"],
                position=Point2D(init_st["shooter"]["x"], init_st["shooter"]["y"]),
                radius=init_st["shooter"].get("radius", 0.6),
            )
            targets = [
                Target(
                    id=t["id"],
                    position=Point2D(t["x"], t["y"]),
                    radius=t.get("radius", 0.8),
                    is_alive=t.get("is_alive", True),
                )
                for t in init_st.get("targets", [])
            ]
            obstacles = []
            for obs in init_st.get("obstacles", []):
                bbox = obs.get("bounding_box", {})
                obs_type = obs.get("type", "box")
                if obs_type == "box":
                    obstacles.append(
                        BoxObstacle(
                            x_min=bbox["x_min"],
                            x_max=bbox["x_max"],
                            y_min=bbox["y_min"],
                            y_max=bbox["y_max"],
                            name=obs.get("name", "obstacle"),
                        )
                    )
                elif obs_type == "circle":
                    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
                    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
                    r = (bbox["x_max"] - bbox["x_min"]) / 2.0
                    obstacles.append(
                        CircleObstacle(center=Point2D(cx, cy), radius=r, name=obs.get("name", "obs"))
                    )

            self.game_state = GameState(
                bounds=bounds, shooter=shooter, targets=targets, obstacles=obstacles
            )
            self.renderer.bounds = self.game_state.bounds

            self._show_attempt(self.current_attempt_idx)

        except Exception as e:
            messagebox.showerror("Error loading JSON", f"Failed to load history file:\n{str(e)}")

    def _show_attempt(self, attempt_idx: int):
        if not self.loaded_session_data or self.total_attempts == 0:
            return

        attempt_idx = max(0, min(attempt_idx, self.total_attempts - 1))
        self.current_attempt_idx = attempt_idx
        history_item = self.loaded_session_data["history"][attempt_idx]

        llm_out = history_item.get("llm_output", {})
        sim_res = history_item.get("simulation_result", {})
        formula = llm_out.get("formula", "")

        # Compute trajectory result for this attempt
        self.game_state.history.clear()
        res = self.game_state.fire_formula(formula)

        # Parse waypoints
        raw_wps = llm_out.get("planned_waypoints", [])
        waypoints = [Point2D(w["x"], w["y"]) for w in raw_wps if "x" in w and "y" in w]

        # Update Navigation Bar
        test_name = self.loaded_session_data.get("test_name", "Test")
        self.attempt_step_lbl.config(
            text=f"📂 {test_name} | Attempt {attempt_idx + 1} of {self.total_attempts}"
        )
        self.prev_btn.config(state=tk.NORMAL if attempt_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if attempt_idx < self.total_attempts - 1 else tk.DISABLED)

        # Update Sidebar Badges
        strategy = llm_out.get("strategy", "N/A")
        self.strategy_lbl.config(text=f"Strategy: {strategy}")

        is_succ = sim_res.get("is_success", False)
        hit_type = sim_res.get("hit_type", "unknown").upper()
        if is_succ:
            self.outcome_lbl.config(text="Status: TARGET HIT! ✅", fg="#22C55E")
        else:
            hit_obs = sim_res.get("hit_obstacle", "")
            obs_str = f" ({hit_obs})" if hit_obs else ""
            self.outcome_lbl.config(text=f"Status: {hit_type}{obs_str} ❌", fg="#EF4444")

        # Update Sidebar Text with full LLM Reasoning & Details
        self.diag_text.delete("1.0", tk.END)
        lines = [
            f"=== Attempt {attempt_idx + 1} Details ===",
            f"Formula: y = {formula}",
            f"Timestamp: {history_item.get('timestamp', 'N/A')}",
            f"",
            f"💡 LLM Reasoning:",
            f"{llm_out.get('reasoning', 'No reasoning provided.')}",
            f"",
            f"📍 Planned Waypoints ({len(waypoints)}):",
        ]
        for i, wp in enumerate(waypoints):
            lines.append(f"  P{i+1}: ({wp.x:.2f}, {wp.y:.2f})")

        lines.extend([
            f"",
            f"🎯 Simulation Feedback:",
            f"  - Hit Type: {hit_type}",
            f"  - Impact Coordinate: {sim_res.get('hit_coordinate', 'None')}",
            f"  - Collided Obstacle: {sim_res.get('hit_obstacle', 'None')}",
            f"  - Distance to Target: {sim_res.get('closest_distance_to_target', 'N/A')}",
        ])
        self.diag_text.insert(tk.END, "\n".join(lines))

        # Render Plot
        attempt_title = f"{test_name} - Attempt {attempt_idx + 1}/{self.total_attempts} (Strategy: {strategy})"
        self.renderer.render(
            self.game_state,
            current_result=res,
            waypoints=waypoints,
            attempt_label=attempt_title,
        )
        self.canvas.draw()

    def _prev_attempt(self):
        if self.playback_mode and self.current_attempt_idx > 0:
            self._show_attempt(self.current_attempt_idx - 1)

    def _next_attempt(self):
        if self.playback_mode and self.current_attempt_idx < self.total_attempts - 1:
            self._show_attempt(self.current_attempt_idx + 1)

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.coord_lbl.config(text=f"Cursor: ({event.xdata:6.2f}, {event.ydata:6.2f})")

    def _on_preset_change(self, event=None):
        self.playback_mode = False
        self.mode_lbl.config(text="[LIVE MODE]", fg="#22C55E")
        self.attempt_step_lbl.config(text="Live Simulation Mode")
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.strategy_lbl.config(text="Strategy: Live Interactive")
        self.outcome_lbl.config(text="Status: Ready", fg="#38BDF8")

        self.current_preset = self.preset_var.get()
        self.game_state = GameState.create_preset(self.current_preset)
        self.renderer.bounds = self.game_state.bounds
        self.refresh_display()

    def _clear_live(self):
        self.game_state.history.clear()
        self.refresh_display()

    def _on_fire(self):
        self.playback_mode = False
        self.mode_lbl.config(text="[LIVE MODE]", fg="#22C55E")
        formula = self.formula_var.get().strip()
        if not formula:
            return

        result = self.game_state.fire_formula(formula)
        self.refresh_display(current_result=result)

    def refresh_display(self, current_result=None):
        self.renderer.render(self.game_state, current_result)
        self.canvas.draw()
        self.diag_text.delete("1.0", tk.END)
        desc = self.game_state.get_text_description() or ""
        self.diag_text.insert(tk.END, str(desc))


def launch():
    root = tk.Tk()
    app = GraphWarApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
