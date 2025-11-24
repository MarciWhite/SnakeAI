# Snake AI with Reinforcement Learning

## About the Application
This is a custom-made, interactive UI application for controlling the training and evaluation process of a Reinforcement Learning (RL) Snake agent. It allows users to set hyperparameters, load/save models, and visualize the training progress in real-time.

## About the AI
The AI is built using **Deep Q-Learning** with **PyTorch**.

* **Algorithm:** Q-learning with Experience Replay.
* **Optimizer:** PyTorch's built-in Adam optimizer.
* **Strategy:** Epsilon-Greedy exploration/exploitation.
    * *Note: While not strictly optimal for all RL environments, this serves as a robust baseline.*
* **Architecture:** Single hidden layer with a changeable neuron count.

### Network Architecture (Layers)
* **Input Layer (12 dimensions):**
    * 3 neurons: Danger detection (Straight, Right, Left).
    * 4 neurons: Apple direction (N, E, S, W).
    * 4 neurons: Current moving direction (N, E, S, W).
    * 1 neuron: Urgency (0-1 scale representing how urgent it is to find an apple; game ends after a set amount of idle steps).
* **Hidden Layer:** 1 layer, 256 neurons by default (customizable).
* **Output Layer:** 3 neurons representing the action for the best predicted outcome (Straight, Right Turn, Left Turn).

## About the Saved Models
Saved models are stored in the `./model/` folder by default (the folder is created if it doesn't exist). The models are weights directly exported from PyTorch.

* **Metadata:** Upon the first save, a `metadata.json` file is generated alongside the model. This includes hyperparameters, high score, and mean score.
* **Auto-save:** The code automatically saves models that achieve a new high score.
* **Managing Files:** You can edit the `.json` reasonably. If you want to delete a saved model, simply delete the model file; the metadata will update automatically upon the next run.
    * *Warning:* If you rename a model file manually, the program will assume the original was deleted unless you also update the `.json` file.

## About Hyperparameters
Reinforcement learning is highly sensitive to parameters. Note that there is very little set in stone; sometimes changing a parameter by a couple of digits can cause drastic shifts in stability.

* **Start Epsilon:** (0 - 1)
    * A percentage that decides if the agent should make a random action or try to predict it.
    * *Typical value:* 0.7 - 0.9.
* **Epsilon Decay:** (Usually around 0.0001)
    * The number subtracted from the current epsilon after every step, making random moves less likely over time.
    * *Too low:* Long learning process.
    * *Too high:* The network stops exploring before it learns the environment.
* **Minimal Epsilon:** (0 - 0.005)
    * Sets a lower limit to epsilon. It is often beneficial to keep this non-zero to maintain slight exploration.
* **Gamma:** (0 - 1)
    * Also known as the **discount rate**. It determines how the network weighs future rewards.
    * *0:* Short-sighted (cares only about immediate rewards).
    * *1:* Long-sighted (cares equally about long and short-term gains).
    * *Recommendation:* > 0.8 for strategy-based games.
* **Learning Rate:** (Usually around 0.001)
    * Determines how much the program tweaks the neurons after each learning step.
    * *Too high:* Causes instability.
    * *Too low:* Makes the training process extremely long.
* **Hidden Size:**
    * Number of neurons in the hidden layer.
    * *Low count:* Won't see patterns above a certain complexity.
    * *High count:* Can overfit (learns complicated patterns but fails to generalize to new states).
* **Memory Size:**
    * Determines how many states the agent remembers. Usually, a higher number is better (limited only by hardware/RAM), though sometimes you may want to limit memory of very old states.
* **Batch Size:**
    * Determines how many states are chosen from memory for each training step.
    * *Typical:* 32-64.
    * *Note:* I have found that a larger batch size (e.g., 1000) works surprisingly well in this environment.

## Evaluation and Visualisation
1.  **Load a Model:** Select a model from your system or choose one from the dropdown in the UI.
2.  **Run Evaluation:** Check the **Eval** box and start the model.
3.  **Simulation:** It will simulate games according to the sample size (default: 1000) and save the results into a `.csv` file in the `./statistics/` folder.
4.  **Graphs:** Run the `visualise_stats.py` script. This reads all `.csv` files in the `./statistics/` folder and saves graph images to a new folder called `./graphs/`.