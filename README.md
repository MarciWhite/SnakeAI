# Snake AI with  reinforcement learning
## About the application
Custom made interactive UI for controlling the training and evaluating process, can be used to set the hyperparameters, load/save models.

## About the AI
- It uses Q-learning with PyTorch.
- Optimizer: PyTorch's built-in Adam optimizer
- Epsilon greedy exploration/exploitation (Note: that it's not optimal for this type of RL)
- Single hidden layer with changeable neuron count

## About the saved models
Saved models are put in the ./model/ folder by default (if it doesn't exist it will create one).
The saved models are the weights directly exported from PyTorch. 
The code auto saves highscoring models. Upon first save a metadata.json file will be generated alongside the saved model, which includes information about the models such as hyperparams, score, meanscore etc.
Feel free to edit this .json reasonably. If you want to delete a saved model you can just delete the model file, the metadata will be updated accordingly upon running the code again. (Note that this also means that after renaming a model will make the program think it was deleted, unless changing the .json file before running the code)

## About hyperparameters/parameters
- Start epsilon value: Number between 0-1, A percentage that decides if it should make a random action or try to predict it. (This is usually around 0.7-0.9 in similar applications)
- Epsilon decay: Number usually around 0.0001, A number that will be subtracted from the current epsilon after every step making it less and less likely that the move will be random. If the number is too low the learning process will be long, if it's too small the neural network won't have enough time to explore the states.
- Minimal epsilon: Number usually around 0-0.005, This will set a lower limit to the epsilon, I have found that sometimes It's beneficial to set it to a non-zero value, but you should experiment with it.
- Gamma: Number between 0-1, It's usually referred to as the discount rate, It determines how the neural network should weigh future rewards. 0 being totally short-sighted, doesn't care about future rewards at all. 1 being It cares equally about long and short term gain. In applications of RL, where strategy is important it should be high. I recommend setting it to something above 0.8 in this case.
- Learning rate: Number usually around 0.001, This number determines how much the program should tweak the neurons after each learning step. If gamma is too much it causes instability, if it's too low it makes the training process really long. It's usually a good idea to have a higher learning rate at the start of the training and make it lower the further we are into training, but I haven't implemented support for this yet.
- Hidden size: Number of neurons in the hidden layer, I encourage you to experiment with this. Lower neuron count means that it won't see patterns above a certain complexity, and higher number means it could overfit, and learn really complicated patterns and will fail when presented with a new state due to lack of generalization.
- Memory size: It determines how much states it should remember. Usually there is no downside of it being a higher number apart from the hardware strain, but sometimes you don't want your model to remember older states.
- Batch size: Determines how many state will be chosen from memory for each training step. It's usually recommended to keep it relatively small, around 32-64 but I have found that it works with 1000 surprisingly well. I encourage you to experiment with this.
- Note that there is very little set in stone in reinforcement learning, sometimes even a couple of digits changing could cause drastic shifts in stability, but there are no wrong options within the set limits.

## Evaluation and visualisation
- You have to load a model either by selecting it from your system or choosing one from the dropdown. Check the eval box and start the model.
- It will simulate the games according to the sample size, which is a 1000 by default, and save the results of them into a .csv file in ./statistics/ folder.
- I have also included a visualise_stats.py script that creates graphs (and saves them as images to a new folder called ./graphs/) from all the .csv file inside the ./statistics/ folder.

## Layers
- Input layer: 12 dimensions: 3 directions for danger, 4 directions for where the apple is, 4 directions for where we are currently moving, number between 0-1 that represents how urgent it is to find an apple (after certain amount of idle steps the game ends)
- Hidden layer: 
- Output layer: Direction for the best predicted outcome (0,1,2)