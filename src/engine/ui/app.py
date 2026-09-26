import sys
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from src.engine.backend.space import Point2D, CoordinateBounds
from src.engine.backend.game_state import GameState
from src.engine.backend.physics import HitType
from src.engine.ui.renderer import ArenaRenderer


class GraphWarApp:
    """
    Interactive 2D Graphwar Simulation GUI.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GraphWars-LLM Simulation Engine")
        self.root.geometry("1100x750")
        self.root.configure(bg="#0F172A")

        # Initialize Game State with default preset
        self.current_preset = "pillar"
        self.game_state = GameState.create_preset(self.current_preset)

        self._build_ui()
        self._setup_plot()
        self.refresh_display()

    def _build_ui(self):
        # Top Header & Scenario Selection
        top_bar = tk.Frame(self.root, bg="#1E293B", pady=8, padx=12)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            top_bar,
            text="⚔️ GraphWars-LLM 2D Simulator",
            font=("Segoe UI", 12, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        title_lbl.pack(side=tk.LEFT, padx=(0, 20))

        preset_lbl = tk.Label(
            top_bar, text="Map Preset:", font=("Segoe UI", 10), fg="#94A3B8", bg="#1E293B"
        )
        preset_lbl.pack(side=tk.LEFT, padx=(0, 5))

        self.preset_var = tk.StringVar(value="pillar")
        presets = ["pillar", "slalom", "bunker", "terrain_valley", "direct_shot"]
        preset_menu = ttk.Combobox(
            top_bar,
            textvariable=self.preset_var,
            values=presets,
            state="readonly",
            width=15,
        )
        preset_menu.pack(side=tk.LEFT, padx=(0, 10))
        preset_menu.bind("<<ComboboxSelected>>", self._on_preset_change)

        reset_btn = tk.Button(
            top_bar,
            text="🔄 Reset Map",
            command=self._reset_map,
            bg="#334155",
            fg="#F8FAFC",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2",
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Coordinate Hover Label
        self.coord_lbl = tk.Label(
            top_bar,
            text="Cursor: (0.00, 0.00)",
            font=("Consolas", 10),
            fg="#38BDF8",
            bg="#1E293B",
        )
        self.coord_lbl.pack(side=tk.RIGHT, padx=10)

        # Main Layout: Left = Plot, Right = Controls & Diagnostics
        main_frame = tk.Frame(self.root, bg="#0F172A")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Canvas Frame
        self.plot_frame = tk.Frame(main_frame, bg="#0F172A")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Sidebar Frame
        sidebar = tk.Frame(main_frame, bg="#1E293B", width=340, padx=12, pady=12)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        sidebar.pack_propagate(False)

        # Sidebar: Formula Input Section
        input_title = tk.Label(
            sidebar,
            text="Mathematical Trajectory Input",
            font=("Segoe UI", 11, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        input_title.pack(anchor=tk.W, pady=(0, 6))

        formula_desc = tk.Label(
            sidebar,
            text="Enter y = f(x) (e.g. sin(0.2*x)*5, 0.02*(x+20)*(x-20)+7)",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#1E293B",
            wraplength=310,
            justify=tk.LEFT,
        )
        formula_desc.pack(anchor=tk.W, pady=(0, 8))

        # Formula Entry Box
        input_row = tk.Frame(sidebar, bg="#1E293B")
        input_row.pack(fill=tk.X, pady=(0, 10))

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

        # Action Buttons
        btn_row = tk.Frame(sidebar, bg="#1E293B")
        btn_row.pack(fill=tk.X, pady=(0, 15))

        fire_btn = tk.Button(
            btn_row,
            text="🚀 Fire Trajectory",
            command=self._on_fire,
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
        )
        fire_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        clear_btn = tk.Button(
            btn_row,
            text="Clear",
            command=self._clear_history,
            font=("Segoe UI", 10),
            bg="#334155",
            fg="#F8FAFC",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
        )
        clear_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Quick Example Formula Buttons
        ex_lbl = tk.Label(
            sidebar, text="Quick Examples:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#1E293B"
        )
        ex_lbl.pack(anchor=tk.W, pady=(5, 4))

        examples_frame = tk.Frame(sidebar, bg="#1E293B")
        examples_frame.pack(fill=tk.X, pady=(0, 15))

        examples = [
            ("Arc Over", "-0.02*(x+20)*(x-20)"),
            ("Sin Wave", "6 * sin(0.15 * x)"),
            ("Low Line", "0.05 * x"),
        ]
        for name, expr in examples:
            btn = tk.Button(
                examples_frame,
                text=name,
                font=("Segoe UI", 8),
                bg="#334155",
                fg="#E2E8F0",
                relief=tk.FLAT,
                command=lambda e=expr: self._set_formula(e),
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Divider
        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, pady=10)

        # Diagnostics & Result Panel
        diag_title = tk.Label(
            sidebar,
            text="📊 Trajectory Diagnostics",
            font=("Segoe UI", 11, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        diag_title.pack(anchor=tk.W, pady=(0, 6))

        self.diag_text = tk.Text(
            sidebar,
            height=13,
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
        
        # Connect mouse motion event for coordinate readout
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.coord_lbl.config(text=f"Cursor: ({event.xdata:6.2f}, {event.ydata:6.2f})")

    def _set_formula(self, formula: str):
        self.formula_var.set(formula)

    def _on_preset_change(self, event=None):
        self.current_preset = self.preset_var.get()
        self.game_state = GameState.create_preset(self.current_preset)
        self.renderer.bounds = self.game_state.bounds
        self.refresh_display()

    def _reset_map(self):
        self.game_state = GameState.create_preset(self.current_preset)
        self.refresh_display()

    def _clear_history(self):
        self.game_state.history.clear()
        self.refresh_display()

    def _on_fire(self):
        formula = self.formula_var.get().strip()
        if not formula:
            return

        result = self.game_state.fire_formula(formula)
        self.refresh_display(current_result=result)

    def refresh_display(self, current_result=None):
        self.renderer.render(self.game_state, current_result)
        self.canvas.draw()
        self._update_diagnostics(current_result)

    def _update_diagnostics(self, current_result=None):
        self.diag_text.delete("1.0", tk.END)
        
        desc = self.game_state.get_text_description()
        self.diag_text.insert(tk.END, desc)


def launch():
    """Launch the interactive GUI application."""
    root = tk.Tk()
    app = GraphWarApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
