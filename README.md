# minitorch

The full minitorch student suite.

## Setup

Requires **Python 3.11** (originally pinned `numba==0.56`/`numpy==1.22` only
support up to Python 3.10, so dependencies were bumped to `numba>=0.57,<0.58`
and `numpy==1.24.4` — the overlapping compatible range, you can compare with previous versions of requirements.txt and
requirements.extra.txt).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip check
```

To access the autograder:

* Module 0: https://classroom.github.com/a/qDYKZff9
* Module 1: https://classroom.github.com/a/6TiImUiy
* Module 2: https://classroom.github.com/a/0ZHJeTA0
* Module 3: https://classroom.github.com/a/U5CMJec1
* Module 4: https://classroom.github.com/a/04QA6HZK
* Quizzes: https://classroom.github.com/a/bGcGc12k

## Task 0.5: Visualization

Manually classified the **Simple** dataset using the following parameters:

- weight_0_0 = -10.00
- weight_1_0 = 0.00
- bias_0 = 5.00

![Simple dataset classification](images/task0_5_simple.png)

## Task 1.5: Training

Trained a small scalar-based neural network (Module 1: Scalar) using the manual autograd/backprop implementation, across
several datasets.

### Simple

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 500
- Dataset: Simple, 50 points

Loss: 2.5\
Final accuracy: ~50/50 correct

![Training on Simple dataset](images/task1_5_simple.png)

### Diag

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 500
- Dataset: Diag, 50 points

Loss: 4.7\
Final accuracy: ~50/50 correct

![Training on Diag dataset](images/task1_5_diag.png)

### Split

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 1500
- Dataset: Split, 50 points

Loss: 4.8\
Final accuracy: ~49/50 correct

![Training on Split dataset](images/task1_5_split.png)

### Xor

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 1500
- Dataset: Xor, 50 points

Loss: 10.2\
Final accuracy: ~47/50 correct

![Training on Xor dataset](images/task1_5_xor.png)

## Task 2.5: Tensor Training

Reimplemented the same three-layer network from Module 1 (Linear → ReLU → Linear → ReLU → Linear → Sigmoid), now using
the tensor-based autograd engine (Module 2) instead of scalars. The model architecture and training logic are identical
to `run_scalar.py` — only the underlying representation changed from individual `Scalar` objects to `Tensor`
operations (`map`/`zip`/`reduce`, broadcasting), which build a much smaller computation graph per forward pass and run
the actual arithmetic through vectorized tensor ops instead of per-element Python calls.

Trained and evaluated on all four datasets, using the Streamlit tensor sandbox (Module 2 → Module 2: Tensor).

### Simple

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 500

Loss: ...
Final accuracy: 50/50 correct
Time per epoch: 0.13 s

![Tensor training on Simple dataset](images/task2_5_simple.png)

### Diag

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 500

Loss: 0.4
Final accuracy: 50/50 correct
Time per epoch: 0.13 s

![Tensor training on Diag dataset](images/task2_5_diag.png)

### Split

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 1500

Loss: 6.3
Final accuracy: 50/50 correct
Time per epoch: 0.13 s

![Tensor training on Split dataset](images/task2_5_split.png)

### Xor

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 1500

Loss: 10.1
Final accuracy: 48/50 correct
Time per epoch: 0.13 s

![Tensor training on Xor dataset](images/task2_5_xor.png)

### Circle

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 1500

Loss: 10.2
Final accuracy: 50/50 correct
Time per epoch: 0.13 s

![Tensor training on Xor dataset](images/task2_5_circle.png)

### Spiral

Parameters used:

- Hidden layer size: 8
- Learning rate: 0.05
- Epochs: 1500

Loss: 34.2
Final accuracy: 27/50 correct
Time per epoch: 0.13 s

![Tensor training on Xor dataset](images/task2_5_spiral.png)