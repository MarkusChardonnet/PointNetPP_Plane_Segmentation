 # Point Cloud Segmentation with PointNet++

## Introduction

The goal of this project is to implement a method of point cloud segmentation. Having a 3D shape made of a point cloud that represents an object or a scene, segmentation is about assigning a label to each point of the cloud, according to the component it is part of. The method we use in this project is PointNet++, which is derived from PointNet, a Deep Learning architecture for analyzing point clouds. First, the general structure of PointNet++ will be presented. Then, the implementation will be explained as well as the training set used to train and test the method. The results will be presented and analyzed in the last part.

## Project Structure

The project is organized into the following files:

- `eval_mode.py`: Script to evaluate the trained model.
- `train_model.py`: Script to train the PointNet++ model.

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/point-cloud-segmentation.git
    cd point-cloud-segmentation
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Evaluation Mode

To evaluate the trained model, use the `eval_mode.py` script:

```bash
python eval_mode.py <split> <element> <PATH_TO_MODEL>
```

<split>: The dataset split to evaluate (e.g., 'train', 'test').
<element>: The element index to evaluate.
<PATH_TO_MODEL>: The path to the trained model checkpoint.


### Training Mode
To train the PointNet++ model, use the train_model.py script:

```bash
python train_model.py <TRAINING_SIZE> <EPOCHS> <START> [PATH_TO_MODEL]
```

<TRAINING_SIZE>: The size of the training dataset.
<EPOCHS>: The number of training epochs.
<START>: The starting epoch (useful for resuming training).
<PATH_TO_MODEL>: (Optional) The path to a pre-trained model checkpoint to resume training.