import numpy as np

# ── Load the dataset ──────────────────────────────────────────
data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)

#  column assignments
survived = data[:, 1]
pclass   = data[:, 2]
age      = data[:, 6]
fare     = data[:, 10]

# Remove NaN values
valid_ages = age[~np.isnan(age)]
valid_fare = fare[~np.isnan(fare)]

# ── Question 1: Average age ───────────────────────────────────
avg_age = np.mean(valid_ages)
print(f"1. Average age of passengers: {avg_age:.1f} years")

# ── Question 2: Survival rate ─────────────────────────────────
survival_rate = np.nanmean(survived) * 100
print(f"2. Survival rate: {survival_rate:.1f}%")

# ── Question 3: Average fare ──────────────────────────────────
avg_fare = np.mean(valid_fare)
print(f"3. Average fare paid: ${avg_fare:.2f}")

# ── Question 4: Oldest and youngest ──────────────────────────
oldest   = np.max(valid_ages)
youngest = np.min(valid_ages)
print(f"4. Oldest passenger: {oldest:.0f} yrs | Youngest: {youngest:.1f} yrs")

# ── Question 5: Passengers per class ─────────────────────────
for c in [1, 2, 3]:
    count = np.sum(pclass == c)
    print(f"5. Class {c}: {count} passengers")