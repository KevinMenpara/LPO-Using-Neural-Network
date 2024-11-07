import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load and preprocess data
def load_and_preprocess_data(filepath):
    """
    This function loads the CSV data, normalizes numeric columns (Frequency, Difficulty, Recency),
    and label encodes the 'Topic' column.
    
    Args:
    - filepath (str): The path to the CSV file containing interview data.
    
    Returns:
    - data (DataFrame): Processed data with normalized features and label encoded target.
    - label_encoder (LabelEncoder): Fitted label encoder for encoding 'Topic'.
    """
    # Load data from the given CSV file
    data = pd.read_csv(filepath)
    
    # Normalize the numeric features (Frequency, Difficulty, Recency) using MinMaxScaler
    scaler = MinMaxScaler()
    data[['Frequency', 'Difficulty', 'Recency']] = scaler.fit_transform(data[['Frequency', 'Difficulty', 'Recency']])
    
    # Label encode the Topic column to make it numeric (as it is categorical)
    label_encoder = LabelEncoder()
    data['Topic'] = label_encoder.fit_transform(data['Topic'])
    
    return data, label_encoder

# Define the Neural Network for Regression (Linear Regression model)
class LearningPathModel(nn.Module):
    """
    A simple neural network model for regression. The model is designed with three layers:
    - Input layer (with input_dim neurons)
    - Hidden layer 1 (64 neurons, ReLU activation)
    - Hidden layer 2 (32 neurons, ReLU activation)
    - Output layer (1 neuron for regression output)
    """
    def __init__(self, input_dim):
        super(LearningPathModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)  # First hidden layer with 64 neurons
        self.layer2 = nn.Linear(64, 32)         # Second hidden layer with 32 neurons
        self.layer3 = nn.Linear(32, 1)          # Output layer with 1 neuron for regression output
        
    def forward(self, x):
        """
        Forward pass through the network. The input data x is passed through each layer
        with ReLU activation for the first two layers and linear activation for the output layer.
        
        Args:
        - x (Tensor): Input data tensor.
        
        Returns:
        - x (Tensor): Predicted output tensor.
        """
        x = torch.relu(self.layer1(x))  # Apply ReLU activation after the first layer
        x = torch.relu(self.layer2(x))  # Apply ReLU activation after the second layer
        return self.layer3(x)           # Output layer with no activation (linear for regression)

# Training function for a single model
def train_model(X_train, y_train, epochs=500):
    """
    This function trains the neural network model using Mean Squared Error (MSE) as the loss function
    and Adam optimizer.
    
    Args:
    - X_train (DataFrame): Input features (Frequency, Difficulty, Recency).
    - y_train (Series): Target variable (Topic).
    - epochs (int): Number of epochs for training (default is 100).
    
    Returns:
    - model (LearningPathModel): The trained model.
    """
    # Get the number of features (input dimension)
    input_dim = X_train.shape[1]
    
    # Initialize the model, loss function, and optimizer
    model = LearningPathModel(input_dim)  # Initialize the neural network model
    criterion = nn.MSELoss()              # Mean Squared Error loss function for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001
    
    # Training loop
    for epoch in range(epochs):
        # Convert the training data to PyTorch tensors
        inputs = torch.tensor(X_train.values, dtype=torch.float32)  # Convert features to tensor
        labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Convert target to tensor
        
        # Zero the gradients of the model parameters
        optimizer.zero_grad()
        
        # Perform forward pass to get predictions
        outputs = model(inputs)
        
        # Calculate the loss (Mean Squared Error)
        loss = criterion(outputs, labels)
        
        # Perform backward pass to compute gradients
        loss.backward()
        
        # Update the model parameters using the optimizer
        optimizer.step()
        
        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model  # Return the trained model

# Train models for each company
def train_for_each_company(data):
    """
    This function trains a separate model for each company in the dataset.
    
    Args:
    - data (DataFrame): The preprocessed data containing the features and labels for training.
    
    Returns:
    - company_models (dict): A dictionary where the key is the company name and the value is the trained model.
    """
    # Get the unique companies from the dataset
    companies = data['Company'].unique()
    
    company_models = {}  # Dictionary to store the trained models for each company
    
    # Loop through each company and train a model
    for company in companies:
        # Filter the data for the current company
        company_data = data[data['Company'] == company]
        
        # Separate the features (X) and target (y) for training
        X = company_data[['Frequency', 'Difficulty', 'Recency']]  # Features: Frequency, Difficulty, Recency
        y = company_data['Topic']  # Target: Topic
        
        # Train the model for the current company
        print(f"Training model for {company}...")
        model = train_model(X, y, epochs=100)  # Train the model
        
        # Store the trained model in the dictionary
        company_models[company] = model
    
    return company_models  # Return the dictionary of trained models

# Function to extract the learned weights and biases and create linear equation
def get_linear_equation(model):
    """
    This function extracts the weights and biases from the trained model and returns them.
    
    Args:
    - model (LearningPathModel): The trained model.
    
    Returns:
    - weights (ndarray): The weights of the first layer.
    - biases (ndarray): The biases of the first layer.
    """
    with torch.no_grad():  # Disable gradient tracking for inference
        # Get the weights and biases of the first layer
        weights = model.layer1.weight.detach().numpy()
        biases = model.layer1.bias.detach().numpy()
        
    return weights, biases  # Return the weights and biases

# Function to plot the linear equation against the real data for each company
def plot_linear_equation_for_company(model, data, company_name, label_encoder):
    """
    This function generates and saves plots of the linear equation for each company, comparing the predicted
    values with the actual data.
    
    Args:
    - model (LearningPathModel): The trained model for the company.
    - data (DataFrame): The preprocessed data containing features and labels.
    - company_name (str): The name of the company for which the plot is generated.
    - label_encoder (LabelEncoder): The label encoder used to encode the 'Topic' column.
    """
    # Filter data for the selected company
    company_data = data[data['Company'] == company_name]
    
    # Create a sub-folder for the company in './plots/' directory
    company_folder = f'./plots/{company_name}'
    if not os.path.exists(company_folder):
        os.makedirs(company_folder)  # Create folder if it doesn't exist

    # Get linear equation (weights and biases) from the model
    weights, biases = get_linear_equation(model)
    
    # Plot for Frequency vs Topic (keeping Difficulty and Recency constant)
    plt.figure(figsize=(10, 6))  # Create a new figure for the plot
    plt.scatter(company_data['Frequency'], company_data['Topic'], color='blue', label='Actual Data')  # Plot actual data

    # Generate a range of frequencies for plotting the regression line
    freq_range = np.linspace(company_data['Frequency'].min(), company_data['Frequency'].max(), 100)
    # Keep Difficulty and Recency constant at their average values
    difficulty_avg = company_data['Difficulty'].mean()
    recency_avg = company_data['Recency'].mean()
    
    # Prepare inputs for prediction
    inputs = np.column_stack((freq_range, np.full_like(freq_range, difficulty_avg), np.full_like(freq_range, recency_avg)))
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)  # Convert inputs to tensor
    predictions = model(inputs_tensor).detach().numpy().flatten()  # Get predictions from the model
    
    # Plot the regression line
    plt.plot(freq_range, predictions, color='red', label='Regression Line')
    plt.xlabel('Frequency')
    plt.ylabel('Predicted Topic')
    plt.title(f'Regression Line for {company_name} (Frequency vs Topic)')
    plt.legend()
    
    # Save the plot in the company's folder
    plt.savefig(f'{company_folder}/{company_name}_linear_equation_frequency_vs_topic.png')
    plt.close()  # Close the plot to avoid memory issues

    # Plot for Recency vs Topic (keeping Difficulty constant)
    plt.figure(figsize=(10, 6))
    plt.scatter(company_data['Recency'], company_data['Topic'], color='blue', label='Actual Data')
    
    # Generate a range of recency values for plotting the line
    recency_range = np.linspace(company_data['Recency'].min(), company_data['Recency'].max(), 100)
    frequency_avg = company_data['Frequency'].mean()
    
    # Prepare inputs for prediction
    inputs = np.column_stack((np.full_like(recency_range, frequency_avg), np.full_like(recency_range, difficulty_avg), recency_range))
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    predictions = model(inputs_tensor).detach().numpy().flatten()
    
    # Plot the regression line
    plt.plot(recency_range, predictions, color='red', label='Regression Line')
    plt.xlabel('Recency')
    plt.ylabel('Predicted Topic')
    plt.title(f'Regression Line for {company_name} (Recency vs Topic)')
    plt.legend()
    
    # Save plot
    plt.savefig(f'{company_folder}/{company_name}_linear_equation_recency_vs_topic.png')
    plt.close()

    # Plot for Difficulty vs Topic (keeping Frequency and Recency constant)
    plt.figure(figsize=(10, 6))
    plt.scatter(company_data['Difficulty'], company_data['Topic'], color='blue', label='Actual Data')

    # Generate a range of difficulty values
    difficulty_range = np.linspace(company_data['Difficulty'].min(), company_data['Difficulty'].max(), 100)
    frequency_avg = company_data['Frequency'].mean()
    recency_avg = company_data['Recency'].mean()
    
    # Prepare inputs for prediction
    inputs = np.column_stack((np.full_like(difficulty_range, frequency_avg), difficulty_range, np.full_like(difficulty_range, recency_avg)))
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    predictions = model(inputs_tensor).detach().numpy().flatten()
    
    # Plot the regression line
    plt.plot(difficulty_range, predictions, color='red', label='Regression Line')
    plt.xlabel('Difficulty')
    plt.ylabel('Predicted Topic')
    plt.title(f'Regression Line for {company_name} (Difficulty vs Topic)')
    plt.legend()
    
    # Save the plot
    plt.savefig(f'{company_folder}/{company_name}_linear_equation_difficulty_vs_topic.png')
    plt.close()


def plot_4d_graph_with_top_5_predictions(data, label_encoder, company_model, company_name):
    """
    This function generates a 3D scatter plot where the color represents the 'Topic' column
    and the axes represent Frequency, Difficulty, and Recency.
    It highlights the top 5 predicted topics for a given company model.
    The plot is saved in the ./plots/ directory.

    Args:
    - data (DataFrame): The preprocessed data containing features and labels.
    - label_encoder (LabelEncoder): The label encoder used to encode the 'Topic' column.
    - company_model (LearningPathModel): The trained model for the company.
    - company_name (str): The name of the company for which the plot is generated.
    """
    # Filter data for the selected company
    company_data = data[data['Company'] == company_name]
    
    # Extract the features and target
    X = company_data[['Frequency', 'Difficulty', 'Recency']].values
    y = company_data['Topic'].values
    
    # Decode the 'Topic' back to original labels (optional)
    topics = label_encoder.inverse_transform(y)
    
    # Get the predicted topics from the model
    inputs_tensor = torch.tensor(X, dtype=torch.float32)  # Convert features to tensor
    predictions = company_model(inputs_tensor).detach().numpy().flatten()  # Get predictions
    predicted_topics = np.round(predictions).astype(int)  # Round predictions to get topic indices
    
    # Get the indices of the top 5 predicted topics
    top_5_indices = np.argsort(predicted_topics)[-5:]  # Get indices of the top 5 topics
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with Frequency, Difficulty, Recency as coordinates, and Topic as color
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=50, marker='o', label='Actual Topics')
    
    # Highlight the top 5 predicted topics with a different color (e.g., red)
    highlighted = ax.scatter(X[top_5_indices, 0], X[top_5_indices, 1], X[top_5_indices, 2], 
                             c=predicted_topics[top_5_indices], cmap='coolwarm', s=100, marker='^', label='Top 5 Predicted Topics', edgecolors='k')
    
    # Labels for the axes
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Difficulty')
    ax.set_zlabel('Recency')
    
    # Color bar for 'Topic'
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Actual Topic')
    
    # Set title for the plot
    ax.set_title(f'4D Visualization with Top 5 Predicted Topics for {company_name} (Frequency, Difficulty, Recency, Topic)')
    
    # Add legend
    ax.legend()
    
    # Ensure the ./plots/ directory exists
    plot_dir = './plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)  # Create the directory if it doesn't exist
    
    # Save the plot in the company's folder under ./plots/
    plot_path = f'{plot_dir}/{company_name}_4d_plot_with_top_5_predictions.png'
    plt.savefig(plot_path)
    
    # Show the plot
    plt.show()

# Main function to run the workflow
def main():
    """
    The main function that drives the workflow: loads the data, trains models for each company,
    and generates plots of the regression equations for each company.
    """
    # Load and preprocess data
    filepath = 'historical_interview_data.csv'  # Path to the data file
    data, label_encoder = load_and_preprocess_data(filepath)
    
    # # Train models for each company
    company_models = train_for_each_company(data)
    
    # Plot and save linear equations for each company
    for company_name, model in company_models.items():
        plot_linear_equation_for_company(model, data, company_name, label_encoder)
    
    # Train models for each company and generate plots with predictions
    companies = data['Company'].unique()
    
    for company_name in companies:
        print(f"Training and plotting for {company_name}...")
        
        # Filter the data for the current company
        company_data = data[data['Company'] == company_name]
        X = company_data[['Frequency', 'Difficulty', 'Recency']]
        y = company_data['Topic']
        
        # Train the model for the current company
        model = train_model(X, y, epochs=100)  # Train the model
        
        # Plot the 4D graph with highlighted top 5 predicted topics and save it
        plot_4d_graph_with_top_5_predictions(data, label_encoder, model, company_name)

# Entry point of the script
if __name__ == "__main__":
    main()  # Run the main function
