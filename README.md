# GNN using Pytorch on Cora Citation Data

This Python script implements a **Graph Convolutional Network (GCN)** using PyTorch on the **Cora dataset**. The Cora dataset is a common benchmark in graph-based learning tasks, particularly citation networks. Below is a detailed explanation of the various components in the code and how they work together:

---

### **1. Libraries and Device Setup**
- **Libraries**: 
  - The script uses PyTorch, NumPy, and SciPy for numerical operations and matrix computations.
  - `torch.nn` and `torch.nn.functional` provide the building blocks for neural networks.
  - **scipy.sparse** is used for sparse matrix representations, which are efficient for graph data.
  
- **Device Setup**:
  - The script checks if CUDA (GPU) or MPS (Apple Silicon GPUs) is available and assigns the appropriate device (`cuda`, `mps`, or `cpu`).

### **2. One-Hot Encoding and Data Loading (`load_data`)**

#### `encode_onehot`:
- Encodes the class labels (e.g., categories of papers in the Cora dataset) as **one-hot vectors**.
  
#### `load_data`:
- **Loads the Cora dataset**: 
  - The dataset consists of a graph where nodes are research papers and edges are citations between them.
  - The `.content` file contains the features for each paper, and the `.cites` file contains the edges representing the citation relationships.
  
- **Feature Matrix**: 
  - The node features are loaded as a sparse matrix (using SciPy).
  
- **Adjacency Matrix**:
  - The graph is represented using an adjacency matrix where edges between nodes are captured. The adjacency matrix is made **symmetric** (undirected graph) and normalized.

- **Train/Test Split**:
  - Divides the nodes into training, validation, and test sets (indices `idx_train`, `idx_val`, `idx_test`).
  
- **Conversion to PyTorch Tensors**:
  - The feature matrix, labels, and adjacency matrix are converted into PyTorch tensors for further processing by the GCN.

---

### **3. Graph Convolution Layer (`GraphConvolution`)**

The `GraphConvolution` class defines the forward pass for a single layer in a GCN. It is based on the original paper ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907).

- **Initialization (`__init__`)**:
  - Takes the number of input features (`in_features`) and output features (`out_features`).
  - Initializes the weight matrix `W` and an optional bias.

- **Forward Pass (`forward`)**:
  - Multiplies the input features by the weight matrix: `support = torch.mm(input, self.weight)`.
  - Multiplies the result with the normalized adjacency matrix (`adj`) to propagate information through the graph: `output = torch.spmm(adj, support)`.
  - Adds bias if it's used.

This layer essentially performs **graph convolution** by transforming the input features and propagating the information across the graph's edges.

---

### **4. GCN Model (`GCN`)**

- The `GCN` class defines the architecture of the GCN, consisting of two graph convolutional layers.

#### **Initialization (`__init__`)**:
- Defines two GCN layers:
  1. `gc1`: Takes input node features and maps them to a hidden layer of size `nhid` (16).
  2. `gc2`: Takes the hidden layer and maps it to the number of output classes (e.g., categories of papers).
  
- Includes a **dropout** layer to avoid overfitting.

#### **Forward Pass (`forward`)**:
- The input features are passed through the first graph convolutional layer (`gc1`), followed by ReLU activation and dropout.
- The output is passed through the second layer (`gc2`) to get the final node classification scores.
- The final output is passed through a **log softmax** function to produce class probabilities for each node.

---

### **5. Training and Testing Functions**

#### **Training (`train`)**:
- The model is trained using **Negative Log Likelihood Loss (NLL Loss)**, which is suitable for classification tasks with `log_softmax` output.
- The optimizer is Adam, which is commonly used for its efficiency in handling large datasets and sparse matrices.
- Each training epoch:
  1. Sets the model to training mode.
  2. Computes the forward pass using the node features and adjacency matrix.
  3. Computes the loss on the training set, backpropagates the gradients, and updates the model weights.
  4. Optionally evaluates the validation set during training to check the model's performance without training it (turns off dropout).

#### **Testing (`test`)**:
- After training, the model is evaluated on the test set to compute the final accuracy and loss.
- The test set accuracy provides a measure of the model's generalization ability.

---

### **6. Running the Model**

- The model is trained for **200 epochs**.
- During each epoch, the loss and accuracy for both the training and validation sets are printed.
- After training, the model is tested on the test set, and the final loss and accuracy are reported.

---

### **Summary of the GCN Workflow**:
1. **Data Preparation**:
   - The Cora dataset is loaded, and node features and adjacency matrices are constructed.
   - The data is split into training, validation, and test sets.

2. **Model Initialization**:
   - A two-layer GCN is initialized with the input feature size, hidden layer size (16), and output size (number of classes).
   - Adam optimizer is used to optimize the model.

3. **Training**:
   - The model is trained using the graph structure to perform node classification. 
   - The training loss is computed with NLL loss, and accuracy is evaluated on both training and validation sets.

4. **Testing**:
   - After training, the model is tested on unseen data (test set), and the final accuracy is computed.

