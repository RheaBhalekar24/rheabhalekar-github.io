import torch
import numpy as np
import matplotlib.pyplot as plt
import  torch.nn as nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

writer = SummaryWriter('runs/control_allocation_experiment')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ThrusterDataset(Dataset):
    def __init__(self, tau, u, sequence_length = 1):
        self.tau = torch.FloatTensor(tau.T)
        self.u = torch.FloatTensor(u.T)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tau) - self.sequence_length + 1

    def __getitem__(self, idx):
        tau_seq = self.tau[idx:idx + self.sequence_length]
        u_seq = self.u[idx:idx + self.sequence_length]
        return tau_seq, u_seq


class ControlAllocator(nn.Module):
    def __init__(self, input_size = 3, hidden_size = 64, output_size = 5):
        super(ControlAllocator, self).__init__()
        dropout_rate = 0.2
        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_dropout1 = nn.Dropout(dropout_rate)
        self.encoder_lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.encoder_dropout2 = nn.Dropout(dropout_rate)
        self.encoder_dense = nn.Linear(hidden_size, 5)
        # self.encoder_dense_dropout = nn.Dropout(dropout_rate)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(5, hidden_size, batch_first= True)
        self.decoder_dropout1 = nn.Dropout(dropout_rate)
        self.decoder_lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first = True)
        self.decoder_dropout2 = nn.Dropout(dropout_rate)
        self.decoder_dense = nn.Linear(hidden_size, 3)
        # self.decoder_dense_dropout = nn.Dropout(dropout_rate)

        # Define the ranges for rescaling
        self.u_min = torch.tensor([-10000, -5000, -180, -5000, -180]).float().to(device)
        self.u_max = torch.tensor([10000, 5000, 180, 5000, 180]).float().to(device)

    def rescale_u(self, u_normalized):
        return 0.5 * (u_normalized + 1) * (self.u_max - self.u_min) + self.u_min

    def forward(self, x):

        # Encoder
        x, _ = self.encoder_lstm1(x)
        x = self.encoder_dropout1(x)
        x, _ = self.encoder_lstm2(x)
        x = self.encoder_dropout2(x)
        u_hat = self.encoder_dense(x[:, -1, :]) # Take the last time step (batch_size, seq_length, hidden_state) -> (batch_size, hidden_state)
        # u_hat = self.encoder_dense_dropout(u_hat)
        u_hat_normalized = torch.tanh(u_hat)

        # Rescale u_hat to original ranges
        u_hat = self.rescale_u(u_hat_normalized)

        # Decoder
        x = u_hat.unsqueeze(1)              # Add a time dimension (batch_size, 1, hidden_state) (batch_size, seq_length, hidden_state) -> (batch_size, hideen_state)
        x, _ = self.decoder_lstm1(x)
        x = self.decoder_dropout1(x)
        x, _ = self.decoder_lstm2(x)
        x = self.decoder_dropout2(x)
        tau_hat = self.decoder_dense(x[:, -1, :])
        # tau_hat = self.decoder_dense_dropout(tau_hat)

        return u_hat, tau_hat

def L0_loss(tau_batch, u_pred):
    u_pred = u_pred.cpu().detach().numpy()
    tau_cmd = calculate_tau(u_pred.T)
    tau_cmd = torch.FloatTensor(tau_cmd.T)
    tau_cmd = tau_cmd.unsqueeze(1)
    tau_cmd =  tau_cmd.to(device)
    return torch.mean((tau_batch - tau_cmd)**2)

def L1_loss(tau_batch, tau_pred):
    return torch.mean((tau_batch - tau_pred)**2)

def L2_loss(u_pred, u_max):
    return torch.sum(torch.clamp(torch.abs(u_pred) - u_max, min=0))

def L3_loss(u_pred, delta_u_max):
    u_shift = torch.roll(u_pred, shifts=1, dims=0)
    return torch.sum(torch.clamp(torch.abs(u_pred - u_shift) - delta_u_max, min=0))

def L4_loss(u_pred):
    F1 = u_pred[:, 0]
    F2 = u_pred[:, 1]
    F3 = u_pred[:, 3]
    return torch.sum(torch.abs(F1)**(3/2) + torch.abs(F2)**(3/2) + torch.abs(F3)**(3/2))

def L5_loss(u_pred, alpha_c_lower, alpha_c_upper):
    alpha2 = u_pred[:, 2]
    alpha3 = u_pred[:, 4]
    return torch.sum(
        torch.logical_and(alpha2 > alpha_c_lower[0], alpha2 < alpha_c_lower[1]).float() +
        torch.logical_and(alpha3 > alpha_c_upper[0], alpha3 < alpha_c_upper[1]).float()
    )

def get_params(model):
    params_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            params_norm += torch.sum(p.pow(2))
    return params_norm


def combined_loss(model, tau_batch, tau_pred, u_pred, u_max, delta_u_max, alpha_c_lower, alpha_c_upper,k):
    lambda1 = 0.001
    l2_reg = lambda1 * get_params(model)
    l0 = L0_loss(tau_batch, u_pred)
    l1 = L1_loss(tau_batch, tau_pred)
    l2 = L2_loss(u_pred, u_max)
    l3 = L3_loss(u_pred, delta_u_max)
    l4 = L4_loss(u_pred)
    l5 = L5_loss(u_pred, alpha_c_lower, alpha_c_upper)

    return k[0]*l0 + k[1]*l1 + k[2]*l2 + k[3]*l3 + k[4]*l4 + k[5]*l5 + l2_reg

# Making sure to create realistic random sequential data
def random_walk(start, min_val, max_val, n_steps):
    walk = [start]
    for _ in range(1, n_steps):
        step = np.random.normal(0, 1)
        next_point = walk[-1] + step
        next_point = np.clip(next_point, min_val, max_val)
        walk.append(next_point)
    return np.array(walk)

def calculate_tau(u):
    # TODO: Add the matrix trasnformation

    F1, F2, alpha2, F3, alpha3 = u
    alpha2 = np.radians(alpha2)
    alpha3 = np.radians(alpha3)

    l1, l2, l3, l4 = -14, 14.5, -2.7, 2.7
    tau_surge = F2 * np.cos(alpha2) + F3 * np.cos(alpha3)
    tau_sway = F1 + F2 * np.sin(alpha2) + F3 * np.sin(alpha3)
    tau_yaw = (l2 * F1 +
               l1 * F2 * np.sin(alpha2) +
               l1 * F3 * np.sin(alpha3) -
               l3 * F2 * np.cos(alpha2) -
               l4 * F3 * np.cos(alpha3))

    return np.array([tau_surge, tau_sway, tau_yaw])


def main():
        
    # Defining ranges
    ranges = {
        'F1' : [-10000, 10000],
        'F2' : [-5000, 5000],
        'F3' : [-5000, 5000],
        'alpha2' : [-180, 180],
        'alpha3' : [-180, 180]
    }

    # Definig the training data samples
    n_samples = 1000000

    data = {}

    for key, (min_value, max_value) in ranges.items():
        start = np.random.uniform(min_value, max_value)
        data[key] = random_walk(start, min_value, max_value, n_samples)

    u = np.vstack([data[key] for key in ['F1', 'F2', 'alpha2', 'F3', 'alpha3']])

    train_size = int(0.8 * n_samples)
    u_train = u[:, :train_size]
    tau_train = calculate_tau(u_train)

    tau_mean = np.mean(tau_train, axis = 1, keepdims = True)
    tau_std = np.std(tau_train, axis = 1, keepdims = True)
    tau_train_standardized = (tau_train - tau_mean) / tau_std

    u_test = u[:, train_size:]
    tau_test = calculate_tau(u_test)
    tau_mean = np.mean(tau_test, axis = 1, keepdims = True)
    tau_std = np.std(tau_test, axis = 1, keepdims = True)
    tau_test_standardized = (tau_test - tau_mean) / tau_std
    
    print(f"Shape of u: {u.shape}")
    print(f"Shape of u_train:", u_train.shape)
    print(f"Shape of tau: {tau_train_standardized.shape}")
    
    # Assuming tau_standardized and u_train are your numpy arrays
    sequence_length = 1
    batch_size = 1024
    # Create datasets
    train_dataset = ThrusterDataset(tau_train_standardized, u_train, sequence_length)
    test_dataset = ThrusterDataset(tau_test_standardized, u_test, sequence_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                            sampler=torch.utils.data.RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    # Testing the daataloader
    for batch_tau, batch_u in train_loader:
        print(f"Tau batch shape: {batch_tau.shape}")
        print(f"U batch shape: {batch_u.shape}")
        break
        
    model = ControlAllocator().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    n_epochs = 100
    u_max = torch.tensor([30000, 60000, 180, 60000, 180]).to(device)  # Values from the paper
    delta_u_max = torch.tensor([1000, 1000, 10, 1000, 10]).to(device) # Values from the paper
    alpha_c_lower = torch.tensor([-100, -80]).to(device)
    alpha_c_upper = torch.tensor([80, 100]).to(device)
    k = torch.tensor([1e0, 1e0, 1e-1, 1e-7, 1e-5, 1e-1]).to(device)

    # Traning loop
    losses = []  # To store the loss after each epoch
    iteration_losses = []
    test_losses = []
    print_freq = 100
    total_iterations = 0

    min_loss = 1000

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for i, (tau_batch, u_batch) in enumerate(train_loader):
            tau_batch = tau_batch.to(device)
            u_batch = u_batch.to(device)

            optimizer.zero_grad()

            u_pred, tau_pred = model(tau_batch)

            loss = combined_loss(model, tau_batch, tau_pred, u_pred, u_max, delta_u_max, alpha_c_lower, alpha_c_upper,k)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            total_iterations += 1

            if total_iterations % print_freq == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch [{epoch+1}/{n_epochs}], Iteration [{total_iterations}], Loss: {loss.item():.4f}")
                #writer.add_scalar('Loss/train', avg_loss, total_iterations)

                iteration_losses.append((total_iterations, avg_loss))


        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)


        # Evaluate on test data
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for tau_batch, u_batch in test_loader:
                tau_batch = tau_batch.to(device)
                u_batch = u_batch.to(device)
                u_pred, tau_pred = model(tau_batch)
                loss = combined_loss(tau_batch, tau_pred, u_pred, u_max, delta_u_max, alpha_c_lower, alpha_c_upper, k)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Log the average losses to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < min_loss:
            torch.save(model.state_dict(), 'control_allocator_model1.pth')
            min_loss = avg_test_loss

    writer.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(11, n_epochs+1), losses[10:], label='Train Loss')
    plt.plot(range(11, n_epochs+1), test_losses[10:], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.savefig('train_test_losses1.png')
    plt.show()

    # Save losses for future use
    np.save('train_losses1.npy', np.array(losses))
    np.save('test_losses.npy1', np.array(test_losses))
    np.save('iteration_losses1.npy', np.array(iteration_losses))
    
    model = ControlAllocator().to(device)
    model.load_state_dict(torch.load('control_allocator_model1.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Get the first 1000 samples from the test dataset
    test_samples = 1000
    test_tau = test_dataset.tau[:test_samples].to(device)
    test_u = test_dataset.u[:test_samples].to(device)

    # Initialize arrays to store individual losses
    l0_losses = np.zeros(test_samples)
    l1_losses = np.zeros(test_samples)
    l2_losses = np.zeros(test_samples)
    l3_losses = np.zeros(test_samples)
    l4_losses = np.zeros(test_samples)
    l5_losses = np.zeros(test_samples)

    # Evaluate the model on each sample
    with torch.no_grad():
        for i in range(test_samples):
            tau_sample = test_tau[i:i+1]
            tau_sample = tau_sample.unsqueeze(0)
            u_true_sample = test_u[i:i+1]
            #u_true_sample = u_true_sample.unsqueeze(0)

            u_pred, tau_pred = model(tau_sample)

            # Calculate individual loss components
            l0 = L0_loss(tau_sample, u_pred)
            l1 = L1_loss(tau_sample, tau_pred)
            l2 = L2_loss(u_pred, u_max)
            l3 = L3_loss(u_pred, delta_u_max)
            l4 = L4_loss(u_pred)
            l5 = L5_loss(u_pred, alpha_c_lower, alpha_c_upper)

            # Store the losses
            l0_losses[i] = l0.item()
            l1_losses[i] = l1.item()
            l2_losses[i] = l2.item()
            l3_losses[i] = l3.item()
            l4_losses[i] = l4.item()
            l5_losses[i] = l5.item()

    # Plot the individual losses
    plt.figure(figsize=(12, 8))
    plt.plot(l0_losses, label='L0: Generalized Force Error')
    plt.plot(l1_losses, label='L1: Decoder Reconstruction Error')
    plt.plot(l2_losses, label='L2: Thruster Command Magnitude')
    plt.plot(l3_losses, label='L3: Rate Changes')
    plt.plot(l4_losses, label='L4: Power Consumption')
    plt.plot(l5_losses, label='L5: Azimuth Sectors')

    plt.xlabel('Sample')
    plt.ylabel('Loss Value')
    plt.title('Individual Loss Components for First 1000 Test Samples')
    plt.legend()
    plt.yscale('log')  # Using log scale as loss values might vary significantly
    plt.grid(True)
    plt.savefig('individual_losses1.png')
    plt.show()

    # Calculate and print average losses
    avg_losses = {
        'L0': np.mean(l0_losses),
        'L1': np.mean(l1_losses),
        'L2': np.mean(l2_losses),
        'L3': np.mean(l3_losses),
        'L4': np.mean(l4_losses),
        'L5': np.mean(l5_losses)
    }

    print("Average losses:")
    for loss_name, avg_value in avg_losses.items():
        print(f"{loss_name}: {avg_value:.6f}")

    # Save the loss data
    np.savez('individual_losses1.npz',
            l0=l0_losses, l1=l1_losses, l2=l2_losses,
            l3=l3_losses, l4=l4_losses, l5=l5_losses)
    
if __name__ == '__main__':
    main()