# LSTM Demand Forecasting & Inventory Optimization

An end-to-end retail inventory management project that combines **LSTM-based demand forecasting** with **inventory optimization**. The system predicts future demand from historical sales data and recommends order quantities to balance inventory holding costs and stockout costs.

## Features

- **Data Analysis:** Explore historical sales data and identify demand patterns.
- **Demand Forecasting:** Train an LSTM model to predict future demand.
- **Baseline Comparison:** Compare LSTM predictions with a simple forecasting method.
- **Inventory Optimization:** Determine order quantities using a cost-based optimization approach.
- **Inventory Simulation:** Simulate inventory operations over a test period.
- **Performance Evaluation:** Measure forecasting accuracy and inventory performance.
- **Reproducible Workflow:** Organize data processing, model training, optimization, and evaluation into a clear pipeline.

## System Architecture

```text
Historical Sales Data
        |
        v
Data Preprocessing & EDA
        |
        v
Time-Series Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
Baseline Model             LSTM Model
        |                      |
        +----------+-----------+
                   |
                   v
            Demand Forecast
                   |
                   v
       Inventory Optimization
       - Current Inventory
       - Holding Cost
       - Stockout Cost
                   |
                   v
        Recommended Order Quantity
                   |
                   v
          Inventory Simulation
                   |
                   v
        Performance Evaluation
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deep Learning | PyTorch |
| Forecast Evaluation | Scikit-learn |
| Optimization | Custom Grid Search |
| Development | Jupyter Notebook, Git |
| Dataset | TODO: Selected retail sales dataset |

## Project Structure

```text
inventory-optimization-lstm/
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and transformed data
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_baseline.ipynb    # Baseline forecasting
│   ├── 03_lstm.ipynb        # LSTM model training and evaluation
│   ├── 04_inventory_optimization.ipynb
│   └── 05_evaluation.ipynb  # Inventory simulation and comparison
├── src/
│   ├── data.py              # Data loading and preprocessing
│   ├── features.py          # Feature engineering and sequences
│   ├── models.py            # Forecasting models
│   ├── train.py             # Model training
│   ├── forecast.py          # Demand prediction
│   ├── inventory.py         # Inventory cost and optimization
│   └── evaluation.py        # Metrics and simulation evaluation
├── reports/
│   └── figures/             # Figures used in the report
├── results/                 # Metrics and experiment outputs
├── tests/                   # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure may change as the project develops. Remove or update entries that are not implemented in the final repository.*

## Requirements

- Python 3.10 or later (TODO: confirm the version used in the final environment).
- Git.
- Jupyter Notebook or JupyterLab.
- A compatible PyTorch installation.
- Sufficient memory to process the selected dataset and train the model.

The project is designed for local development. A GPU is optional; a CPU can be used for a small dataset and a compact LSTM model.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd inventory-optimization-lstm
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
```

### 4. Prepare the dataset

Download the selected retail sales dataset from its official source and place the required files in `data/raw/`.

Example dataset sources:

- [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data)
- [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
- [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

Choose one primary dataset and follow its data usage terms. The exact file names and preprocessing steps depend on the selected dataset.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. `01_eda.ipynb`
2. `02_baseline.ipynb`
3. `03_lstm.ipynb`
4. `04_inventory_optimization.ipynb`
5. `05_evaluation.ipynb`

## Quick Start

The following example illustrates the basic inventory cost calculation. It is a standalone example and does not require a trained LSTM model.

```python
def calculate_inventory_cost(
    inventory,
    order_quantity,
    demand,
    holding_cost,
    stockout_cost,
):
    available = inventory + order_quantity

    ending_inventory = max(available - demand, 0)
    unmet_demand = max(demand - available, 0)

    holding = ending_inventory * holding_cost
    stockout = unmet_demand * stockout_cost
    total_cost = holding + stockout

    return {
        "ending_inventory": ending_inventory,
        "unmet_demand": unmet_demand,
        "holding_cost": holding,
        "stockout_cost": stockout,
        "total_cost": total_cost,
    }


result = calculate_inventory_cost(
    inventory=20,
    order_quantity=10,
    demand=25,
    holding_cost=1,
    stockout_cost=10,
)

print(result)
```

This example assumes that newly ordered products are available within the modeled period and that unmet demand is treated as lost sales rather than backorders. These assumptions should be adjusted if the final inventory model uses different rules.

## Methodology

### 1. Demand Forecasting

Historical sales are processed into time-series sequences using a sliding window. For example, a 30-day window uses the previous 30 days to predict demand for the next day.

A baseline model is implemented first, followed by an LSTM model. Both models are evaluated on the same time-based test period.

### 2. Inventory Optimization

The inventory policy uses predicted demand, current inventory, and cost parameters to determine the order quantity.

For a simplified single-period setting:

\[
C(Q)=h\max(I+Q-\hat D,0)
+p\max(\hat D-I-Q,0)
\]

Where:

- \(Q\): Order quantity.
- \(I\): Current inventory.
- \(\hat D\): Predicted demand.
- \(h\): Holding cost per unit.
- \(p\): Stockout cost per unmet unit.
- \(C(Q)\): Estimated inventory cost.

Grid search evaluates feasible order quantities and selects the quantity with the lowest estimated cost. This is a simplified decision model; it does not automatically account for demand uncertainty or replenishment lead time.

### 3. Evaluation

The project evaluates both prediction accuracy and inventory performance.

**Forecasting metrics**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Symmetric Mean Absolute Percentage Error (sMAPE), when appropriate

**Inventory metrics**

- Total inventory cost
- Holding cost
- Stockout cost
- Stockout rate
- Service level
- Average inventory

The inventory simulation uses actual demand from the test period to evaluate the decisions made by each policy. Actual future demand must not be used to make the original ordering decision.

## Configuration

The initial version uses parameters defined in the notebooks or Python modules. Important parameters include:

| Parameter | Description |
|---|---|
| `LOOKBACK_WINDOW` | Number of historical days used for forecasting |
| `BATCH_SIZE` | Number of sequences processed per training batch |
| `LEARNING_RATE` | LSTM optimizer learning rate |
| `EPOCHS` | Maximum number of training epochs |
| `INITIAL_INVENTORY` | Starting inventory for the simulation |
| `HOLDING_COST` | Cost per unit of ending inventory |
| `STOCKOUT_COST` | Cost per unit of unmet demand |
| `MAX_ORDER_QUANTITY` | Upper bound of the grid search, if configured |

These are conceptual parameter names; their exact implementation may differ. Record the final values used in each experiment.

No environment variables or `.env` file are required for the basic local version unless additional configuration is introduced.

## Running Tests

If unit tests are implemented in `tests/`, run:

```bash
python -m unittest discover -s tests
```

Tests should cover inventory balance, zero demand, insufficient inventory, excess inventory, cost calculation, and order-quantity optimization.

## Results

**Status:** TODO — Update after running the final experiments.

Add the actual results from the same test period:

| Metric | Baseline | LSTM |
|---|---:|---:|
| MAE | TODO | TODO |
| RMSE | TODO | TODO |
| Total inventory cost | TODO | TODO |
| Stockout cost | TODO | TODO |
| Holding cost | TODO | TODO |
| Service level | TODO | TODO |

The values must come from completed experiments. Forecasting performance and inventory performance should be discussed separately because a more accurate forecast does not always produce a lower inventory cost.

## Limitations

- Recorded sales may not represent true demand when products are out of stock.
- Holding and stockout costs may need to be assumed if they are not provided by the dataset.
- The initial inventory model simplifies replenishment and may not include lead time.
- A point forecast does not fully represent demand uncertainty.
- Results may depend on the selected dataset, test period, and cost assumptions.

## Roadmap

- [ ] Complete data analysis and preprocessing.
- [ ] Implement and evaluate a baseline forecasting model.
- [ ] Train and evaluate the LSTM model.
- [ ] Implement inventory cost calculation and grid search.
- [ ] Simulate inventory operations and compare strategies.
- [ ] Complete the final report and reproducible instructions.
- [ ] Optional: build a Streamlit dashboard.
- [ ] Optional: add lead time or demand uncertainty to the inventory model.

## License

TODO: Choose a license before publishing the repository. For example, the MIT License may be suitable for an educational project if its terms meet your needs. Add a `LICENSE` file and ensure the dataset's own usage terms are respected.

## Acknowledgements

- The authors and maintainers of the selected retail sales dataset.
- The PyTorch, Pandas, NumPy, and Scikit-learn communities.
- Educational resources on time-series forecasting and inventory management.

## Contact and Support

For questions or bug reports, please open an issue in this repository.

**Author:** TODO: Add your name  
**Project type:** Individual learning project — Deep Learning and Optimization