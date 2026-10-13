# 20261_ProjectI_202416766
- The problem find: (X_t is other problem in day t) (DL Part)
$$D_{t+1}=f(D_t,D_{t-1},...,X_t)$$
- Pipeline:
 Historical sales
       │
       ▼
 Data preprocessing
       │
       ▼
   LSTM model
       │
       ▼
 Demand forecast
       │
       ▼
 Inventory state
       │
       ▼
 Optimization
       │
       ▼
Recommended order quantity
       │
       ▼
 Business decision
- We may have: 
  - uncertainty trong forecast
  - lead time
  - safety stock
  - holding cost
  - stockout cost
  - maximum storage capacity
  - minimum order quantity


- Objective function:
$$C(Q)=C_h max(I_t+Q-D,0)+C_s max(D-I_t-Q,0)$$
- Data processing:
  - Make sequence (sliding window)
- Feature Engineering: more details soon
- LSTM: more details soon (use PyTorch)
- Loss function: MSE, MAE, Huber Loss
  - Forecasting metric
- Can use more method like: Navie, Moving AVG, Linear Regression, XGBoost
- Optimization
  - Advanced: Sophisticated Optimization
  - Add more constraints
  - We can change C_s and C_h for experiment -> Visualization
- Evaltuation:
  - Forecasting: MAE, RMSE, MAPE, sMAPE
  - Inventory: Total cost, stockout rate, avg inventory, service level, holding cost, stockout cost
  - May be use Streamlit
- RECOMMENDED STRUCTURE:
inventory-optimization-lstm/
│
├── README.md
│
├── data/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_lstm.ipynb
│   ├── 04_inventory_optimization.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── forecast.py
│   ├── inventory.py
│   └── evaluation.py
│
├── app/
│   └── app.py
│
├── requirements.txt
│
└── README.md

- Data after processing:
  - 1 day wwe can have 5 fearures: sales, price, promotion, day_of_week, holiday. FOCUS LATER
  - Should not have inventory in LSTM (leakage) -> in Optimization layer
- Recommend pipeline:
              FORECASTING
                   │
sales ─────────────┤
price ─────────────┤
promotion ─────────┤
calendar ──────────┤
                   ↓
                 LSTM
                   ↓
             Demand forecast
                   │
                   ↓
              OPTIMIZATION
                   │
inventory ─────────┤
holding cost ──────┤
stockout cost ─────┤
capacity ──────────┤
                   ↓
             Order quantity

- OVERALL:


                 RAW DATA
                     │
                     ▼
              Data Cleaning
                     │
                     ▼
              Data Aggregation
                     │
                     ▼
              Feature Engineering
                     │
                     ▼
              Time-based Split
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Train Data            Test Data
          │
          ▼
       Scaling
          │
          ▼
    Sliding Window
          │
          ▼
 X = [samples, 30, features]
 y = [samples, 1]
          │
          ▼
         LSTM
          │
          ▼
   Demand Forecast
          │
          ▼
 Inventory Optimizer
          │
          ▼
 Recommended Order