import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QGroupBox,      # NEW
    QTextEdit,       # NEW
    QFileDialog
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from SiPM_helper import (
    get_column_number,
    extract_SiPM_data,
    gauss_fit,
    gauss_function,
    rebin_xy
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SiPM Analysis Tool")
        self.resize(1400, 900)

        self.x_data = None
        self.y_data = None

        self.selected_x = None
        self.selected_y = None

        # =========================
        # MULTI-DATA MANAGEMENT
        # =========================
        self.datasets = []
        self.fits = []
        self.frozen = False

        self.color_cycle = ["blue", "green", "orange", "purple", "brown", "cyan"]

        self.build_ui()

    # ==================================================
    # UI
    # ==================================================
    def build_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        controls = QGridLayout()

        controls.addWidget(QLabel("Bias"), 0, 0)
        self.bias_input = QLineEdit("40.0")
        controls.addWidget(self.bias_input, 0, 1)

        controls.addWidget(QLabel("Gain"), 1, 0)
        self.gain_input = QLineEdit("33")
        controls.addWidget(self.gain_input, 1, 1)

        controls.addWidget(QLabel("DAQ"), 2, 0)
        self.daq_combo = QComboBox()
        self.daq_combo.addItems(["1", "2"])
        controls.addWidget(self.daq_combo, 2, 1)

        controls.addWidget(QLabel("Gain Type"), 3, 0)
        self.gain_type_combo = QComboBox()
        self.gain_type_combo.addItems(["HG", "LG"])
        controls.addWidget(self.gain_type_combo, 3, 1)

        controls.addWidget(QLabel("CITIROC"), 4, 0)
        self.citiroc_combo = QComboBox()
        self.citiroc_combo.addItems(["A", "B", "C", "D", "E"])
        controls.addWidget(self.citiroc_combo, 4, 1)

        controls.addWidget(QLabel("Channel"), 5, 0)
        self.channel_input = QLineEdit("0")
        controls.addWidget(self.channel_input, 5, 1)

        controls.addWidget(QLabel("Rebinning:"), 6, 0)
        self.rebinning_input = QLineEdit("5")
        controls.addWidget(self.rebinning_input, 6, 1)

        main_layout.addLayout(controls)

        # ==================================================
        # BUTTONS
        # ==================================================
        buttons = QHBoxLayout()

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        buttons.addWidget(self.load_button)

        self.fit_button = QPushButton("Gaussian Fit")
        self.fit_button.clicked.connect(self.fit_selected_region)
        buttons.addWidget(self.fit_button)

        self.freeze_button = QPushButton("Freeze Dataset")
        self.freeze_button.clicked.connect(self.freeze_dataset)
        buttons.addWidget(self.freeze_button)

        self.clear_fit_button = QPushButton("Clear")
        self.clear_fit_button.clicked.connect(self.clear_fits)
        buttons.addWidget(self.clear_fit_button)

        self.reset_button = QPushButton("Reset Zoom")
        self.reset_button.clicked.connect(self.reset_zoom)
        buttons.addWidget(self.reset_button)

        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        buttons.addWidget(self.save_button)

        main_layout.addLayout(buttons)

        # ==================================================
        # FIT BOX (MODIFICATO)
        # ==================================================
        self.fit_box = QGroupBox("Fit Results")
        self.fit_box.setCheckable(True)
        self.fit_box.setChecked(False)
        self.fit_box.setMaximumHeight(100)
        
        fit_layout = QVBoxLayout()
        self.fit_text = QTextEdit()
        self.fit_text.setReadOnly(True)
        self.fit_text.setMinimumHeight(70)
        self.fit_text.hide()   # nascosto all'avvio

        fit_layout.addWidget(self.fit_text)

        self.fit_box.setLayout(fit_layout)

        self.fit_box.toggled.connect(self.fit_text.setVisible)

        main_layout.addWidget(self.fit_box)

        # ==================================================
        # Plot
        # ==================================================
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("ADC Counts")

        main_layout.addWidget(self.canvas)

        # ==================================================
        # Span selector
        # ==================================================
        self.span = SpanSelector(
            self.ax,
            self.on_region_selected,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0, facecolor="tab:gray")
        )

    # ==================================================
    # FILE PATH
    # ==================================================
    def get_file_path(self, bias, gain):
        return f"./data/{bias}/gain{gain}/data.csv"

    # ==================================================
    # LOAD DATA
    # ==================================================
    def load_data(self):

        self.clear_plot()

        try:
            bias = self.bias_input.text().strip()
            gain = self.gain_input.text().strip()

            daq = int(self.daq_combo.currentText())
            gain_type = self.gain_type_combo.currentText()
            citiroc = self.citiroc_combo.currentText()
            channel = int(self.channel_input.text())
            rebinning = int(self.rebinning_input.text())
            file_path = self.get_file_path(bias, gain)

            column_number = get_column_number(
                DAQ=daq,
                gain_type=gain_type,
                CITIROC=citiroc,
                channel=channel
            )

            x, y = extract_SiPM_data(
                file_path=file_path,
                column_number=column_number,
                one_based_column=False
            )

            x, y = x, y

            self.x_data, self.y_data = x, y

            self.plot_data(x, y)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ==================================================
    # SAVE PLOT
    # ==================================================
    def save_plot(self):
        if not self.datasets:
            QMessageBox.warning(self, "Warning", "No data to save.")
            return

        bias = self.bias_input.text().strip()
        gain = self.gain_input.text().strip()

        # Asking to the user where to save the plot and with which name, suggesting a default name based on bias and gain
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            f"plot_bias_{bias}_gain_{gain}.png",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )

        if not file_name:
            return

        try:
            self.figure.savefig(file_name, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Saved", f"Plot saved as {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot: {str(e)}")

    # ==================================================
    # PLOT
    # ==================================================
    def plot_data(self, x, y):

        color = self.color_cycle[
            len(self.datasets) % len(self.color_cycle)
        ]

        line, = self.ax.plot(
            x,
            y,
            lw=1,
            color=color,
            label=f"Bias: {self.bias_input.text().strip()} | Gain: {self.gain_input.text().strip()}"
        )

        label = f"Bias: {self.bias_input.text().strip()} | Gain: {self.gain_input.text().strip()}"

        self.datasets.append((x, y, label, line))

        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("ADC Counts")

        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='best')

        self.canvas.draw()

    # ==================================================
    # CLEAR PLOT
    # ==================================================
    def clear_plot(self):

        if self.frozen:
            self.frozen = False
            return

        self.ax.clear()
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("ADC Counts")

        self.canvas.draw_idle()

    # ==================================================
    # FREEZE
    # ==================================================
    def freeze_dataset(self):

        if self.x_data is None:
            return

        self.frozen = True

        QMessageBox.information(

            
            self,
            "Frozen",
            "Dataset frozen. You can now load another dataset."
        )

    # ==================================================
    # CLEAR FITS
    # ==================================================
    def clear_fits(self):

        self.fits.clear()

        lines_to_remove = []

        for line in self.ax.get_lines():
            if line.get_color() == "red":
                lines_to_remove.append(line)

        for line in lines_to_remove:
            line.remove()

        self.update_fit_label()
        self.canvas.draw_idle()
        self.ax.clear()  # Clear all lines and reset axes
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("ADC Counts")

    # ==================================================
    # REGION SELECTION
    # ==================================================
    def on_region_selected(self, xmin, xmax):

        if self.x_data is None:
            return

        mask = (self.x_data >= xmin) & (self.x_data <= xmax)

        self.selected_x = self.x_data[mask]
        self.selected_y = self.y_data[mask]

        self.ax.set_xlim(xmin, xmax)

        if np.any(mask):
            y_min = np.min(self.y_data[mask])
            y_max = np.max(self.y_data[mask])
            margin = 0.1 * (y_max - y_min if y_max > y_min else 1)

            self.ax.set_ylim(y_min - margin, y_max + margin)

        self.canvas.draw_idle()

    # ==================================================
    # RESET ZOOM
    # ==================================================
    def reset_zoom(self):

        if self.x_data is None:
            return

        self.ax.set_xlim(np.min(self.x_data), np.max(self.x_data))
        self.ax.set_ylim(np.min(self.y_data), np.max(self.y_data))

        self.canvas.draw_idle()

    # ==================================================
    # FIT
    # ==================================================
    def fit_selected_region(self):

        if self.selected_x is None:
            QMessageBox.warning(self, "Warning", "Select a region first.")
            return

        sy = np.sqrt(self.selected_y)

        popt, perr = gauss_fit(
            self.selected_x,
            self.selected_y,
            sy
        )

        fit_x = np.linspace(
            np.min(self.selected_x),
            np.max(self.selected_x),
            1000
        )

        fit_y = gauss_function(fit_x, *popt)

        # Plotting with the same 
        self.ax.plot(fit_x, fit_y, "r", lw=2, label = f"Gauss B{self.bias_input.text().strip()}G{self.gain_input.text().strip()} $\mu={popt[1]:.1f} +- {perr[1]:.1f}$", color = "darkred")

        self.fits.append({
            "A": (popt[0], perr[0]),
            "mu": (popt[1], perr[1]),
            "sigma": (popt[2], perr[2])
        })

        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='best')

        self.update_fit_label()
        self.canvas.draw()

    # ==================================================
    # FIT DISPLAY (MODIFICATO)
    # ==================================================
    def update_fit_label(self):

        if not self.fits:
            self.fit_text.setText("N/A")
            return

        text = ""

        for i, f in enumerate(self.fits):

            text += (
                f"[Fit {i+1}]\n"
                f"A = {f['A'][0]:.2f} ± {f['A'][1]:.2f}\n"
                f"μ = {f['mu'][0]:.2f} ± {f['mu'][1]:.2f}\n"
                f"σ = {f['sigma'][0]:.2f} ± {f['sigma'][1]:.2f}\n"
                f"----------------------\n"
            )

        self.fit_text.setText(text)


# ==================================================
# MAIN
# ==================================================
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()