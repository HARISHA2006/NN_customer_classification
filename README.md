# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![Screenshot 2025-03-17 112856](https://github.com/user-attachments/assets/bc5d3d4e-be3b-48f6-808f-fdc2fa1144d4)

## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.
### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.
### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).


### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.

## PROGRAM

### Name: HARISHA S
### Register Number: 212223040063

```
class PeopleClassifier(nn.Module):
    def _init_(self, input_size):
        super(PeopleClassifier, self)._init_()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

```
```
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```
# Training Loop

```
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```
## Dataset Information

![Screenshot 2025-03-17 113110](https://github.com/user-attachments/assets/876b68a3-3a95-4f72-84db-a2f05312f01f)


## OUTPUT

## TESTCASE
![Screenshot 2025-03-22 082710](https://github.com/user-attachments/assets/9f19ef23-6718-427a-b04e-788123424398)



### Confusion Matrix
![Screenshot 2025-03-17 113723](https://github.com/user-attachments/assets/16cb2ebf-bc43-41d7-89f8-3d889d742d04)



### Classification Report

![Screenshot 2025-03-17 114350](https://github.com/user-attachments/assets/4580ede2-3f5c-4a46-9c5c-802ed61ce9bb)



### New Sample Data Prediction

![Screenshot 2025-03-22 082811](https://github.com/user-attachments/assets/9954fa68-6689-4827-b05f-a6a29f65c178)

## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
