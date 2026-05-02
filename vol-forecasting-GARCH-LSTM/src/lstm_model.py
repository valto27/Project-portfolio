import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import time
import copy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

n_splits = 5

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, 'data', 'processed')
metrics_dir = os.path.join(base_dir, 'results', 'metrics')
models_dir = os.path.join(base_dir, 'models')
prediction_dir = os.path.join(base_dir, 'results', 'predictions')
for d in [processed_dir, metrics_dir, models_dir, prediction_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

def load_merge_data():
    df_processed = pd.read_csv(os.path.join(processed_dir, "processed_data.csv"), index_col=0, parse_dates=True)
    df_garch = pd.read_csv(os.path.join(processed_dir, "garch_variances.csv"), index_col=0, parse_dates=True)
    
    df_merged = df_processed.merge(df_garch, left_index=True, right_index=True, how='inner')
    df_merged = df_merged.dropna()
    return df_merged

def create_sequences(df, window_size=20):
    features = df[['Log_Returns', 'garch_conditional_vol']].values 
    targets = df['Realized_Variance'].values

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(features[i:i+window_size])
        y.append(targets[i + window_size])
    
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    return X_tensor, Y_tensor

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] 
        out = self.fc(last_out)
        return self.softplus(out)

def train_model():
    df_merged = load_merge_data()
    window_size = 20
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold_results = []
    start_time_total = time.time()

    for fold, (train_index, test_index) in enumerate(tscv.split(df_merged)):
        torch.manual_seed(42)
        np.random.seed(42)

        full_train_df = df_merged.iloc[train_index]
        val_size = int(len(full_train_df) * 0.1)
        
        train_df = full_train_df.iloc[:-val_size]
        val_df = full_train_df.iloc[-(val_size + window_size):]

        X_train, y_train = create_sequences(train_df, window_size)
        X_val, y_val = create_sequences(val_df, window_size)

        
        start_test_idx = max(0, test_index[0] - window_size)
        test_df_extended = df_merged.iloc[start_test_idx : test_index[-1] + 1]
        X_test, y_test = create_sequences(test_df_extended, window_size)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
        
        model = VolatilityLSTM()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        epochs = 200
        patience = 10
        best_val_loss = float('inf')
        counter = 0

        best_model_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            
            if epoch % 10 == 0:
                print(f"Fold {fold+1} | Epoch {epoch} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f"Fold {fold+1} | Early stopping all'epoca {epoch}")
                    break
        
        model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy().flatten()
            y_true = y_test.numpy().flatten()
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            fold_results.append({'Fold': fold + 1, 'RMSE': rmse, 'MAE': mae})
            print(f"Fold {fold+1} completato — RMSE: {rmse:.6f} | MAE: {mae:.6f}")

        if fold == n_splits-1:
            pred_df = pd.DataFrame({
                'y_true': y_true,
                'y_pred_lstm': y_pred
            }, index=df_merged.index[test_index][-len(y_true):])
            pred_df.to_csv(os.path.join(prediction_dir, f"lstm_predictions_fold_{fold + 1}.csv"), index=True)

    total_time = (time.time() - start_time_total) / 60
    
    df_metrics = pd.DataFrame(fold_results)
    mean_metrics = pd.DataFrame([{'Fold': 'Average', 
                                  'RMSE': df_metrics['RMSE'].mean(), 
                                  'MAE': df_metrics['MAE'].mean()}])
    df_metrics = pd.concat([df_metrics, mean_metrics], ignore_index=True)
    
    df_metrics.to_csv(os.path.join(metrics_dir, "lstm_metrics.csv"), index=False)
    torch.save(best_model_state, os.path.join(models_dir, "lstm_volatility_model.pth"))

    print("-" * 40)
    print(f"Training Time Totale: {total_time:.2f} minuti")
    print(f"Metriche salvate in: {metrics_dir}")
    print(f"Modello salvato in: {models_dir}")

if __name__ == "__main__":
    train_model()