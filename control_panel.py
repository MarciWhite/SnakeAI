import json
import os
from datetime import datetime
from multiprocessing import Process
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
import sys


def load_model_metadata(load_file_name):
    folder = "./model/"
    metadata_file = os.path.join(folder, "metadata.json")
    if not os.path.exists(metadata_file):
        print("No metadata.json found. Cannot load model.")
        return

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    # load the model
    if load_file_name == "highest":
        model_metadata = max(metadata["models"], key=lambda x: x["score"])
    elif load_file_name == "latest":
        model_metadata = max(metadata["models"], key=lambda x: x["timestamp"])
    else:
        model_metadata = next((x for x in metadata["models"] if x["file"] == load_file_name), None)

    if model_metadata is None:
        print(f"Metadata for {load_file_name} not found")
    print(3)
    return model_metadata


class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake AI Control Panel")
        self.setGeometry(100, 100, 400, 600)
        self.selected_file = None
        self.model_labels = []
        self.init_ui()


    def init_ui(self):
        self.main_layout = QVBoxLayout()

        # --- Neural Network Settings ---
        nn_header = QLabel("Neural Network Settings")
        nn_header.setAlignment(Qt.AlignCenter)
        nn_header.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.main_layout.addWidget(nn_header)

        # learning_rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setPrefix("Learning Rate: ")
        self.main_layout.addWidget(self.lr_spin)

        # gamma
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_spin.setValue(0.8)
        self.gamma_spin.setPrefix("Gamma: ")
        self.main_layout.addWidget(self.gamma_spin)

        # epsilon_start
        self.epsilon_start_spin = QDoubleSpinBox()
        self.epsilon_start_spin.setRange(0.0, 1.0)
        self.epsilon_start_spin.setSingleStep(0.01)
        self.epsilon_start_spin.setValue(0.95)
        self.epsilon_start_spin.setPrefix("Epsilon Start: ")
        self.main_layout.addWidget(self.epsilon_start_spin)

        # epsilon_min
        self.epsilon_min_spin = QDoubleSpinBox()
        self.epsilon_min_spin.setRange(0.0, 1.0)
        self.epsilon_min_spin.setSingleStep(0.01)
        self.epsilon_min_spin.setValue(0.0)
        self.epsilon_min_spin.setPrefix("Epsilon Min: ")
        self.main_layout.addWidget(self.epsilon_min_spin)

        # epsilon_decay
        self.epsilon_decay_spin = QDoubleSpinBox()
        self.epsilon_decay_spin.setRange(0.0, 0.1)
        self.epsilon_decay_spin.setSingleStep(0.001)
        self.epsilon_decay_spin.setValue(0.01)
        self.epsilon_decay_spin.setPrefix("Epsilon Decay: ")
        self.main_layout.addWidget(self.epsilon_decay_spin)

        # batch_size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10_000)
        self.batch_size_spin.setValue(1000)
        self.batch_size_spin.setPrefix("Batch Size: ")
        self.main_layout.addWidget(self.batch_size_spin)

        # max_memory
        self.max_memory_spin = QSpinBox()
        self.max_memory_spin.setRange(1000, 1_000_000)
        self.max_memory_spin.setValue(100_000)
        self.max_memory_spin.setPrefix("Max Memory: ")
        self.main_layout.addWidget(self.max_memory_spin)

        # hidden_size
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(1, 2048)
        self.hidden_size_spin.setValue(256)
        self.hidden_size_spin.setPrefix("Hidden Size: ")
        self.main_layout.addWidget(self.hidden_size_spin)

        # --- Game Settings ---
        game_header = QLabel("Game Settings")
        game_header.setAlignment(Qt.AlignCenter)
        game_header.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.main_layout.addWidget(game_header)

        # speed
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 500)
        self.speed_spin.setValue(50)
        self.speed_spin.setPrefix("Speed: ")
        self.main_layout.addWidget(self.speed_spin)

        # width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 2000)
        self.width_spin.setValue(640)
        self.width_spin.setPrefix("Width: ")
        self.main_layout.addWidget(self.width_spin)

        # height
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 2000)
        self.height_spin.setValue(480)
        self.height_spin.setPrefix("Height: ")
        self.main_layout.addWidget(self.height_spin)

        # block_size
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(5, 100)
        self.block_size_spin.setValue(20)
        self.block_size_spin.setPrefix("Block Size: ")
        self.main_layout.addWidget(self.block_size_spin)

        # snake_start_size
        self.snake_start_spin = QSpinBox()
        self.snake_start_spin.setRange(1, 50)
        self.snake_start_spin.setValue(3)
        self.snake_start_spin.setPrefix("Snake Start Size: ")
        self.main_layout.addWidget(self.snake_start_spin)

        # hard_boundary
        self.hard_boundary_ck = QCheckBox("Hard Boundary")
        self.hard_boundary_ck.setChecked(True)
        self.main_layout.addWidget(self.hard_boundary_ck)

        # Render checkbox
        self.render_ck = QCheckBox("Render")
        self.render_ck.setChecked(True)
        self.main_layout.addWidget(self.render_ck)

        # Buttons
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        self.main_layout.addWidget(self.load_button)

        self.file_label = QLabel("No file selected")
        self.main_layout.addWidget(self.file_label)

        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.main_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.main_layout.addWidget(self.stop_button)

        self.setLayout(self.main_layout)

    def clear_model_labels(self):
        """Remove existing model info labels."""
        for label in self.model_labels:
            self.layout.removeWidget(label)
            label.deleteLater()
        self.model_labels.clear()

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if file_path:
            file_name = os.path.basename(file_path)
            self.file_label.setText(file_name)
            self.selected_file = file_path

            metadata_file = file_name
            self.clear_model_labels()
            model_metadata = load_model_metadata(metadata_file)
            self.show_model_info(model_metadata)


    def show_model_info(self, metadata):
        """Show metadata labels when a model is loaded."""
        timestamp = metadata.get("timestamp", "unknown")
        # Convert "20251113_172625" → readable format
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            readable_time = dt.strftime("%b %d, %Y – %H:%M")
        except Exception:
            readable_time = timestamp

        info = {
            "Games Played": metadata["model_settings"].get("num_game", "N/A"),
            "Score": metadata.get("score", "N/A"),
            "Mean Score": metadata.get("mean_score", "N/A"),
            "Last Trained": readable_time
        }

        for key, value in info.items():
            label = QLabel(f"{key}: {value}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 14px; color: #333;")
            self.main_layout.addWidget(label)
            self.model_labels.append(label)

    def start_training(self):
        # Collect all settings into a dict
        settings = {
            "learning_rate": self.lr_spin.value(),
            "gamma": self.gamma_spin.value(),
            "epsilon_start": self.epsilon_start_spin.value(),
            "epsilon_min": self.epsilon_min_spin.value(),
            "epsilon_decay": self.epsilon_decay_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "max_memory": self.max_memory_spin.value(),
            "hidden_size": self.hidden_size_spin.value(),
            "speed": self.speed_spin.value(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "block_size": self.block_size_spin.value(),
            "snake_start_size": self.snake_start_spin.value(),
            "hard_boundary": self.hard_boundary_ck.isChecked(),
            "render": self.render_ck.isChecked()
        }
        print("Starting training with settings:", settings)
        # Here you would start your AI training using these settings

    def stop_training(self):
        print("Stop training clicked")
# Function to run PyQt app
def run_gui():
    app = QApplication(sys.argv)
    window = ControlPanel()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    gui_process = Process(target=run_gui)
    gui_process.start()

    # Here you would run your Pygame Snake loop in the main process


    gui_process.join()
