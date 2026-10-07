- If forecast D = 80, it not 100%, we can develop in 2 version (censored demand)
LSTM
 ↓
Point forecast
 ↓
Optimization

or

LSTM
 ↓
Demand distribution / uncertainty
 ↓
Inventory optimization  (NOT RECOMMEND)

- Demand can be predict as probability (35 20%, 50 50%, 70 30%)
$$E[C(Q)]=\sum P(D=d)C(Q,d)$$

- In future we can use Transformer, RL, Bayesian Optimization