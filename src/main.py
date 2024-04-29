from models import create_3d_cnn
from models import create_pointnet_model
from models import create_gnn

# =============================== Convolutional Neural Network ===============================
"""
    For a 3D CNN, we ahve to make sure that the LiDAR data is voxelized, 
    which just means converting the point cloud to a 3D grid.

    For preprocessing we can use voxel grid. Each voxel in the grid can be
    binary (occupied/unoccupied), or we can use other values like point 
    density or average intesity. The input shape of the network just has to 
    match the voxel grids dimensions (depth, heigh, width, channels)
"""
def cnn():
    
    # Code goes here...

# =============================== PointNet Neural  ===============================
"""
    PointNet is awesome, like we talked about it handles raw point clouds, 
    making it highly suitable for our data.

    For preprocessing we'll just need to normalize the point cloud data. For
    example, centering and scaling. We could also do random sampling or cropping 
    to standardize the number of points.

    INput Shape: (x,y,z,maybe intensity?)
"""
def pointnet():
    # Code goes here...

# =============================== Graph Neural Network ===============================
"""

"""
# GNN Example
def gnn():
    # Parameters
    num_features = 3
    hidden_channels = 16
    num_classes = 10

    # Load Data
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())

    # Initialize Model
    model = create_gnn(dataset.num_node_features, hidden_channels, dataset.num_classes)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = dataset[0].to(device)

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the Model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluating the Model
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    print(f'Accuracy: {accuracy:.4f}')

# =============================== Main ===============================
def main():
    # cnn()
    # pointnet()
    gnn()

if __name__ == "__main__":
    main()
