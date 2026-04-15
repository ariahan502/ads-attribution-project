# Ads Attribution & Uplift Modeling 🚀

## 📌 Project Overview

This project simulates a real-world advertising system to optimize **conversion prediction, attribution, and budget allocation**.

Starting from raw user interaction logs, we built an end-to-end pipeline including:

- CTR prediction
- Feature engineering
- Multi-touch attribution
- Uplift (causal) modeling

The goal is to move beyond simple click prediction and answer:

👉 _"Which users should we target to maximize incremental conversions?"_

---

## 🧠 Key Components

### 1. CTR Prediction

- Built baseline Logistic Regression model
- Improved performance using XGBoost
- Feature engineering:

  - campaign-level CTR encoding
  - user behavior features
  - temporal features

📈 Result:

- Logistic Regression AUC: ~0.62
- XGBoost AUC: ~0.75

---

### 2. Feature Engineering

- Campaign CTR encoding (mean encoding)
- Log transformations (cost, CPO)
- Sequential behavior features:

  - time since last click
  - click position

- Categorical features (campaign, categories)

---

### 3. Attribution Modeling

#### Last-touch attribution

- Assigns full credit to the final interaction

#### Multi-touch attribution

- Uses dataset-provided attribution weights

#### Linear attribution

- Evenly distributes credit across all touches

📊 Insight:

- Last-touch tends to over-credit late-stage campaigns
- Multi-touch reveals earlier funnel contributions

---

### 4. Ranking System

We simulate ad ranking using:

```python
score = predicted_ctr × value (CPO)
```

This mimics real-world ad auction logic (e.g., eCPM / expected value ranking).

---

### 5. Uplift Modeling (Causal Inference) ⭐

Instead of predicting:
👉 "Who will convert?"

We predict:
👉 "Who will convert _because of the ad_?"

#### Approach:

- Define treatment: `click = 1`
- Split data:

  - Treated group
  - Control group

- Train two models:

  - P(conversion | treated)
  - P(conversion | control)

#### Uplift:

```
uplift = P_treated - P_control
```

---

## 📊 Results

### CTR vs Uplift

- Correlation ≈ 0
  👉 CTR ≠ true incremental value

---

### Conversion Lift

| Group               | Conversion Rate |
| ------------------- | --------------- |
| Top 10% (by uplift) | ~22%            |
| Bottom 10%          | ~0.2%           |

👉 Massive improvement in targeting efficiency

---

## 💡 Business Impact

This project demonstrates how to:

- Avoid wasting budget on users who would convert anyway
- Identify high-impact users for ads
- Improve ROI through causal modeling
- Move from prediction → decision optimization

---

## 🛠 Tech Stack

- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 📁 Project Structure

```
ads-attribution-project/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ctr_model.ipynb
│
├── data/
├── scripts/
├── README.md
```

---

## 🚀 Future Improvements

- Doubly Robust / Meta-learners (T-Learner, X-Learner)
- Uplift tree models
- Real-time bidding simulation
- Budget allocation optimization

---

## 🎯 Key Takeaway

👉 Traditional CTR models optimize for clicks
👉 Uplift models optimize for **true causal impact**

This is critical for:

- Ads ranking
- Growth strategy
- Marketing ROI optimization
