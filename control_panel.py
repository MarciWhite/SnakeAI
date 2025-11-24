import csv
import json
import os

import numpy as np
import torch

from agent import Agent
from game import SnakeGame

from datetime import datetime
from multiprocessing import Process, Event, Queue
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
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
    return model_metadata


# noinspection PyUnresolvedReferences
class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.main_layout = None
        self.ai_process = None
        self.stop_event = Event()
        self.save_event = Event()
        self.update_queue = Queue() # For communicating live statistics
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(100)  # check every 100ms

        self.setWindowTitle("Snake AI Control Panel")
        self.setGeometry(100, 100, 400, 600)
        self.selected_file = None
        self.model_labels = []

        self.main_layout = QVBoxLayout()
        self.init_ui()
        self.setLayout(self.main_layout)
        self.menu = "Start"

    def check_queue(self):
        if self.menu == "Train":
            while not self.update_queue.empty():
                info = self.update_queue.get()
                # Game info
                try:
                    self.score_label.setText(f"Last Score: {info['last_score']}")
                    self.mean_label.setText(f"Mean Score: {info['mean_score']:.2f}")
                    self.game_label.setText(f"Game: {info['num_games']}")
                    self.highscore_label.setText(f"Highscore: {info['model_highscore']}")
                    self.hotstreak_label.setText(f"Hot streak: {info['hot_streak']}")

                    # Agent stats
                    self.epsilon_label.setText(f"Current Epsilon: {info.get('current_epsilon', 0.0):.3f}")
                    self.memory_label.setText(
                        f"Memory: {info.get('memory_filled', 0)} / {info.get('max_memory', 0)}")
                    self.batch_label.setText(f"Batch Size: {info.get('batch_size', 0)}")
                    self.lr_label.setText(f"Learning Rate: {info.get('learning_rate', 0.0)}")
                    self.gamma_label.setText(f"Gamma: {info.get('gamma', 0.0)}")
                    self.hidden_label.setText(f"Hidden Size: {info.get('hidden_size', 256)}")
                    self.epsilon_params_label.setText(
                        f"Epsilon Params: start={info.get('epsilon_start', 0.0):.2f}, "
                        f"min={info.get('epsilon_min', 0.0):.2f}, "
                        f"decay={info.get('epsilon_decay', 0.00002):.6f}"
                    )
                except Exception as e:
                    print(e)

    def init_ui(self):
        self.menu = "Start"
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
        self.epsilon_min_spin.setDecimals(3)
        self.epsilon_min_spin.setPrefix("Epsilon Min: ")
        self.main_layout.addWidget(self.epsilon_min_spin)

        # epsilon_decay
        self.epsilon_decay_spin = QDoubleSpinBox()
        self.epsilon_decay_spin.setRange(0.0, 0.1)
        self.epsilon_decay_spin.setSingleStep(0.00001)
        self.epsilon_decay_spin.setValue(0.00002)
        self.epsilon_decay_spin.setDecimals(6)
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

        # Render
        self.render_ck = QCheckBox("Render")
        self.render_ck.setChecked(True)
        self.main_layout.addWidget(self.render_ck)

        # Statistics
        self.stats_ck = QCheckBox("Log Statistics in .csv")
        self.stats_ck.setChecked(False)
        self.main_layout.addWidget(self.stats_ck)

        #Evaluate or Training mode
        self.eval_ck = QCheckBox("Evaluate")
        self.eval_ck.setChecked(False)
        self.main_layout.addWidget(self.eval_ck)

        # Buttons
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        self.main_layout.addWidget(self.load_button)

        self.model_combo = QComboBox()
        self.model_combo.addItem("-- Select Model --")
        self.model_combo.addItems(["Highest Score", "Last Trained"])
        self.main_layout.addWidget(self.model_combo)
        self.model_combo.currentIndexChanged.connect(self.handle_selection_change)

        self.file_label = QLabel("No file selected")
        self.main_layout.addWidget(self.file_label)

        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.main_layout.addWidget(self.start_button)

    def handle_selection_change(self, index):
        if index == 1:
            self.selected_file = "highest"
        elif index == 2:
            self.selected_file = "latest"
        else:
            self.selected_file = None

        # Change labels
        if self.selected_file is not None:
            model_metadata = load_model_metadata(self.selected_file)
            self.selected_file = model_metadata["file"]
            self.file_label.setText(model_metadata.get("file","No model selected"))
            self.show_model_info(model_metadata)
        else:
            self.file_label.setText("No model selected")




    def clear_model_labels(self):
        """Remove existing model info labels."""
        for label in self.model_labels:
            try:
                self.main_layout.removeWidget(label)
                label.deleteLater()
            except RuntimeError as ex:
                pass
                # print("Label already deleted")
        self.model_labels.clear()

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)

            # If it's a widget, delete it
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

            # If it's a sub-layout, clear it recursively
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    self.clear_layout(sub_layout)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if file_path:
            file_name = os.path.basename(file_path)
            self.file_label.setText(file_name)
            self.selected_file = file_path

            metadata_file = file_name
            #self.clear_model_labels()
            model_metadata = load_model_metadata(metadata_file)
            self.show_model_info(model_metadata)


    def save_model(self):
        if self.ai_process and self.ai_process.is_alive():
            print("Save requested")
            self.save_event.set()

    def show_model_info(self, metadata):
        """Show metadata labels when a model is loaded."""
        # Display settings
        self.toggle_architecture_inputs(False)
        model_settings = metadata.get("model_settings")
        game_settings = metadata.get("game_settings")
        self.lr_spin.setValue(model_settings['learning_rate'])
        self.gamma_spin.setValue(model_settings['gamma'])
        self.epsilon_start_spin.setValue(model_settings['epsilon_start'])
        self.epsilon_min_spin.setValue(model_settings['epsilon_min'])
        self.epsilon_decay_spin.setValue(model_settings['epsilon_decay'])
        self.batch_size_spin.setValue(model_settings['batch_size'])
        self.max_memory_spin.setValue(model_settings['max_memory'])
        self.hidden_size_spin.setValue(model_settings['hidden_size'])

        self.speed_spin.setValue(game_settings['speed'])
        self.width_spin.setValue(game_settings['width'])
        self.height_spin.setValue(game_settings['height'])
        self.block_size_spin.setValue(game_settings['block_size'])
        self.snake_start_spin.setValue(game_settings['snake_start_size'])
        self.hard_boundary_ck.setChecked(game_settings['hard_boundary'])

        # Display other stats
        self.clear_model_labels()
        timestamp = metadata.get("timestamp", "unknown")
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

    def toggle_architecture_inputs(self, enable: bool):
        self.hidden_size_spin.setEnabled(enable)
        self.gamma_spin.setEnabled(enable)
        self.hard_boundary_ck.setEnabled(enable)

    def start_training(self):
        # Collect all settings into a dict
        settings = {
            "model_settings": {
                "learning_rate": self.lr_spin.value(),
                "gamma": self.gamma_spin.value(),
                "epsilon_start": self.epsilon_start_spin.value(),
                "epsilon_min": self.epsilon_min_spin.value(),
                "epsilon_decay": self.epsilon_decay_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "max_memory": self.max_memory_spin.value(),
                "hidden_size": self.hidden_size_spin.value(),
                "eval": self.eval_ck.isChecked()
            },
            "game_settings": {
                "speed": self.speed_spin.value(),
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "block_size": self.block_size_spin.value(),
                "snake_start_size": self.snake_start_spin.value(),
                "hard_boundary": self.hard_boundary_ck.isChecked(),
                "render": self.render_ck.isChecked(),
                "stats": self.stats_ck.isChecked()
            }
        }

        # Stops training if one is running
        self.stop_training()
        self.stop_event.clear()
        self.save_event.clear()

        print(self.selected_file)
        self.ai_process = Process(
            target=run_ai,
            args=(self.stop_event, self.save_event, self.update_queue, self.selected_file, settings)
        )
        self.ai_process.start()
        self.selected_file = None

        #print("Starting training with settings:", settings)
        eval_mode = False
        try:
            eval_mode = settings["model_settings"]["eval"]
        except KeyError as e:
            print("Key Error")
        self.clear_layout(self.main_layout)
        self.training_menu(eval_mode)


    def training_menu(self, eval_mode=False):
        self.menu = "Train"

        # --- Game Stats Header ---
        self.game_stats_header = QLabel("Game Stats")
        self.game_stats_header.setAlignment(Qt.AlignCenter)
        self.game_stats_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.main_layout.addWidget(self.game_stats_header)

        # Game stats labels
        self.highscore_label = QLabel("Highscore: 0")
        self.score_label = QLabel("Score: 0")
        self.mean_label = QLabel("Mean Score: 0.0")
        self.game_label = QLabel("Game: 0")
        self.hotstreak_label = QLabel("Hotstreak: 0")
        for lbl in [self.score_label, self.mean_label, self.game_label, self.hotstreak_label, self.highscore_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 12px; color: #333;")
            self.main_layout.addWidget(lbl)

        # --- Agent Stats Header ---
        self.agent_stats_header = QLabel("Agent Stats")
        self.agent_stats_header.setAlignment(Qt.AlignCenter)
        self.agent_stats_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.main_layout.addWidget(self.agent_stats_header)

        # Agent stats labels
        self.epsilon_label = QLabel("Epsilon: 0.0")
        self.memory_label = QLabel("Memory: 0 / 0")
        self.batch_label = QLabel("Batch Size: 0")
        self.lr_label = QLabel("Learning Rate: 0.0")
        self.gamma_label = QLabel("Gamma: 0.0")
        self.hidden_label = QLabel("Hidden Size: 0")
        self.epsilon_params_label = QLabel("Epsilon Params: start=0.0, min=0.0, decay=0.0")

        for lbl in [self.epsilon_label, self.memory_label, self.batch_label,
                    self.lr_label, self.gamma_label, self.hidden_label, self.epsilon_params_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 12px; color: #333;")
            self.main_layout.addWidget(lbl)

        if not eval_mode:
            self.save_button = QPushButton("Save Model")
            self.save_button.clicked.connect(self.save_model)
            self.main_layout.addWidget(self.save_button)

            self.stop_button = QPushButton("Stop Training")
        else:
            self.stop_button = QPushButton("Stop Evaluating")
        self.stop_button.clicked.connect(self.stop_training)
        self.main_layout.addWidget(self.stop_button)


    def stop_training(self):
        self.menu = "Stopping"
        if self.ai_process and self.ai_process.is_alive():
            print("Stopping training...")
            self.stop_event.set()
            self.ai_process.join(timeout=2)

            if self.ai_process.is_alive():
                self.ai_process.terminate()
                print("AI training process terminated successfully")

        self.clear_layout(self.main_layout)
        self.init_ui()

    def alert(self, title, message):
        QMessageBox.information(
            self,
            title,
            message
        )



def run_gui():
    app = QApplication(sys.argv)
    window = ControlPanel()
    window.show()
    sys.exit(app.exec_())


def add_to_csv(load_file_name, games, score, acc_reward):
    folder = "./statistics/"
    if load_file_name is not None:
        csv_path = os.path.join(folder,f"statistics_{load_file_name.replace(".pth","")}.csv")
        os.makedirs(folder, exist_ok=True)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # write header first if file was newly created
            if write_header:
                writer.writerow(["episode", "score","accumulated_reward"])

            # append the new score
            writer.writerow([games, score,acc_reward])
    else:
        print("While evaluating you must load a model.")



SAMPLE_SIZE = 1000
def run_ai(stop_event, save_event, update_queue, load_file_name=None, settings=None):
    folder = "./model/"
    metadata_file = os.path.join(folder, "metadata.json")
    highscore = 0


    if not os.path.exists(metadata_file):
        print("No metadata.json found.")
    else:
        # Update metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)


        updated_models = [x for x in metadata["models"] if os.path.exists(x["file_path"])]
        if updated_models != metadata["models"]:
            print("Changes detected in metadata")
            metadata["models"] = updated_models
            metadata["highscore"] = max(updated_models,key=lambda a: a["score"])["score"]

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)

        highscore = metadata["highscore"]


    scores = []
    model_highscore = 0
    hot_streak = 0
    episodes = 0
    if load_file_name is None:
        if settings is None:
            agent = Agent()
            game = SnakeGame(ai=True,render=False)
        else:
            agent = Agent(settings=settings["model_settings"])
            game = SnakeGame(ai=True, render=settings["game_settings"]["render"],settings=settings["game_settings"])
    else:
        load_file_name = os.path.basename(load_file_name)
        model_metadata = load_model_metadata(load_file_name)

        if settings is None:
            print("Something went wrong. Settings can't be None when loading model.")
            return

        agent = Agent(model_metadata=model_metadata,settings=settings["model_settings"])
        game = SnakeGame(ai=True, render=settings["game_settings"]["render"], settings=settings["game_settings"])

        if not settings["model_settings"]["eval"]: scores = [model_metadata["mean_score"] for _ in range(agent.n_games)]
        model_highscore = model_metadata["score"]


        print(f"Training started with settings: {agent.export_settings()} and game settings: {game.export_settings()}")
    try:
        if settings["model_settings"]["eval"]:
            agent.epsilon = 0
            agent.epsilon_min = 0
            agent.model.eval()
        else:
            agent.model.train()
    except Exception as e:
        agent.model.train() #Key Error


    while not stop_event.is_set():

        # Evaluate
        if settings["model_settings"]["eval"]:
            game.reset()
            agent.reset_steps()
            done = False
            score = 0
            accumulated_reward = 0
            with torch.no_grad():  # No gradient computation
                while not done:
                    state = agent.get_state(game)
                    move = agent.get_action(state)
                    reward, done, score = game.play_step(action=move, agent=agent)
                    accumulated_reward += reward
            scores.append(score)
            episodes+=1
            add_to_csv(load_file_name,episodes,score, accumulated_reward)
            print(f"Eval episode {episodes} complete.")

            if score > model_highscore:
                model_highscore = score

            mean_score = np.mean(scores) if scores else 0
            if score > mean_score:
                hot_streak+=1
            else:
                hot_streak = 0
            # Send back live stats
            info = agent.export_stats()
            info["model_highscore"] = model_highscore
            info["last_score"] = score
            info["mean_score"] = mean_score
            info["hot_streak"] = hot_streak
            info["num_games"] = episodes
            if update_queue is not None:
                update_queue.put(info)

            if episodes == SAMPLE_SIZE:
                print("Sampling complete. Exiting...")
                stop_event.set()
            continue



        state_old = agent.get_state(game)

        #get move
        move = agent.get_action(state_old)

        reward, done, score = game.play_step(action=move, agent=agent)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, move, reward, state_new, done)
        agent.remember(state_old, move, reward, state_new, done)

        if save_event.is_set():
            print("Save event triggered - Saving model...")
            agent.model.save(
                model_highscore,
                np.mean(scores) if scores else 0,
                agent.export_settings(),
                game.export_settings()
            )
            save_event.clear()
            print("Model saved.")

        # Game is over
        if done:
            print(f"Game #{agent.n_games} finished with score {score}, current mean score: {np.mean(scores) if scores else 0:.2f} Idle death: {agent.steps_left == 0}")
            game.reset()
            agent.reset_steps()
            scores.append(score)
            agent.n_games += 1
            agent.train_long_memory()
            if score > highscore:
                highscore = score
                agent.model.save(score,np.mean(scores) if scores else 0,agent.export_settings(),game.export_settings())

            if score > model_highscore:
                model_highscore = score
            mean_score = np.mean(scores) if scores else 0
            if score > mean_score:
                hot_streak+=1
            else:
                hot_streak = 0
            # Send back live stats
            info = agent.export_stats()
            info["model_highscore"] = model_highscore
            info["last_score"] = score
            info["mean_score"] = mean_score
            info["hot_streak"] = hot_streak
            if update_queue is not None:
                update_queue.put(info)

            if settings["game_settings"]["stats"]:
                add_to_csv(load_file_name, agent.n_games, score, 0)

    # Stopping gracefully comes here
    print("Graceful stop")
    game.stop()





if __name__ == "__main__":
    gui_process = Process(target=run_gui)
    gui_process.start()
    gui_process.join()
