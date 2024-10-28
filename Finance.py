#this py file is a conversion of google notebook that is also available as finance.ipynb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate sample daily expense data for a user
daily_expenses = {
    'date': pd.date_range(start='2024-09-01', end='2024-09-30'),
    'amount': np.random.uniform(30, 80, size=30)
}
df_expenses = pd.DataFrame(daily_expenses)

# Preprocess data for LSTM
scaler = MinMaxScaler()
data = scaler.fit_transform(df_expenses[['amount']])

# Prepare sequences for LSTM (e.g., using last 5 days to predict next day)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(data, seq_length)

# Define the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Predict the next day's expense
predicted = model.predict(np.expand_dims(data[-seq_length:], axis=0))
predicted_expense = scaler.inverse_transform(predicted)[0][0]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df_expenses['date'], df_expenses['amount'], label='Actual Daily Expenses', color='blue')
plt.scatter(df_expenses['date'].iloc[-1] + pd.Timedelta(days=1), predicted_expense, color='red', label='Predicted Next Day Expense')
plt.title("Daily Expenses and Predicted Next Day Expense")
plt.xlabel("Date")
plt.ylabel("Expense Amount")
plt.legend()
plt.show()

print(f"Predicted Next Day Expense: {predicted_expense:.2f}")


import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Define the demo dataset with labeled transactions
transaction_texts = [
    "Bought groceries at Walmart",             # groceries
    "Netflix subscription",                    # entertainment
    "Paid apartment rent",                     # rent
    "Grocery shopping at Target",              # groceries
    "Spotify premium subscription",            # entertainment
    "Paid for groceries at Whole Foods",       # groceries
    "Purchased tickets for a concert",         # entertainment
    "Transferred money for rent",              # rent
    "Bought movie tickets for Saturday night", # entertainment
    "Fresh vegetables and fruits",             # groceries
]

transaction_labels = [0, 1, 2, 0, 1, 0, 1, 2, 1, 0]  # Labels: 0 = groceries, 1 = entertainment, 2 = rent

# Label to category mapping for later reference
label_map = {0: "groceries", 1: "entertainment", 2: "rent"}

# Step 2: Create a custom Dataset class for loading and processing data
class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create train and test datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(transaction_texts, transaction_labels, test_size=0.2, random_state=42)
train_dataset = TransactionDataset(train_texts, train_labels, tokenizer)
val_dataset = TransactionDataset(val_texts, val_labels, tokenizer)

# Step 3: Load BERT model for classification with 3 categories
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Step 4: Define the TrainingArguments and Trainer for fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()

# Step 5: Evaluate the model on validation data
eval_result = trainer.evaluate()
print(f"Evaluation result: {eval_result}")

# Step 6: Test predictions with fine-tuned model
test_data = [
    "Bought some groceries",
    "Paid rent for November",
    "Enjoyed a movie on Netflix"
]
inputs = tokenizer(test_data, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
predicted_categories = [label_map[pred.item()] for pred in predictions]

# Displaying predictions
for transaction, category in zip(test_data, predicted_categories):
    print(f"Transaction: {transaction} | Predicted Category: {category}")


import numpy as np
import pandas as pd

data = {
    'day': list(range(1, 31)),  # 30-day period
    'current_balance': np.random.randint(500, 1500, 30),  # Random current balance amounts
    'target_budget': [1000] * 30,  # Set a consistent target budget for simplicity
    'days_passed': list(range(1, 31)),
    'remaining_budget': np.random.randint(200, 800, 30)  # Random remaining budget amounts
}

# Convert to DataFrame
spending_summary_df = pd.DataFrame(data)

# Initialize Q-table for RL model
num_days = 30  # Period in days
states = 100  # Discretized states for balance levels
actions = 3  # 0: "Stay on track", 1: "Increase savings", 2: "Adjust spending"
q_table = np.zeros((states, num_days, actions))

# Define parameters for the Q-learning model
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.5
exploration_decay = 0.95
min_exploration_rate = 0.1

# Training Loop over the spending summary dataset
for index, row in spending_summary_df.iterrows():
    # State representation based on spending balance and days passed
    target_budget = row['target_budget']
    current_balance_state = min(int(row['current_balance'] // (target_budget / states)), states - 1)
    day_state = row['days_passed'] - 1

    # Action selection
    if np.random.uniform(0, 1) < exploration_rate:
        action = np.random.choice(actions)  # Explore random advice
    else:
        action = np.argmax(q_table[current_balance_state, day_state, :])  # Exploit learned policy

    # Simulate reward based on action
    if action == 0:  # "Stay on track"
        reward = 1 if row['current_balance'] <= target_budget else -1
    elif action == 1:  # "Increase savings"
        reward = 2 if row['remaining_budget'] >= target_budget * 0.5 else -2
    elif action == 2:  # "Adjust spending"
        reward = 1 if row['remaining_budget'] >= target_budget * 0.2 else -1

    # Update Q-table using Q-learning formula
    next_day_state = min(day_state + 1, num_days - 1)
    next_balance_state = min(int(row['current_balance'] // (target_budget / states)), states - 1)
    best_future_action = np.argmax(q_table[next_balance_state, next_day_state, :])

    q_table[current_balance_state, day_state, action] += learning_rate * (
        reward + discount_factor * q_table[next_balance_state, next_day_state, best_future_action] -
        q_table[current_balance_state, day_state, action]
    )

    # Decay exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Final Advice Generation
advice_map = {0: "Stay on track", 1: "Increase savings", 2: "Adjust spending"}
predictions = []
for day_state in range(num_days):
    # Aggregate advice per day
    recommended_action = np.argmax(q_table[:, day_state, :].sum(axis=0))
    predictions.append(advice_map[recommended_action])

# Display the predicted advice for each day
print("\nPredicted Advice for Each Day:")
for day, advice in enumerate(predictions, 1):
    print(f"Day {day}: {advice}")
