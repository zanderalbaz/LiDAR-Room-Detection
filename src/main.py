from models import create_3d_cnn
from models import create_pointnet_model
from models import create_gnn

def cnn():
    # Code goes here...

def pointnet():
    # Code goes here...


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

def main():
    # cnn()
    # pointnet()
    gnn()

if __name__ == "__main__":
    main()
