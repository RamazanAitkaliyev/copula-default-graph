"""
Synthetic Data Generator for Copula Default Graph

Generates a network of 1000 persons across 3 cities with:
- Clear risk archetypes (low/medium/high risk individuals)
- High-risk groups (tightly connected clusters)
- Contagion bridges (well-connected individuals across cities)
- Realistic transaction patterns
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CityConfig:
    """Configuration for a city."""
    name: str
    population: int
    base_risk_level: str  # 'low', 'medium', 'high'
    economic_strength: float  # 0-1, higher = stronger economy


# Default city configuration
DEFAULT_CITIES = [
    CityConfig("Alpha", 400, "low", 0.8),      # Financial hub, diverse, low risk
    CityConfig("Beta", 350, "medium", 0.5),    # Industrial, some stress
    CityConfig("Gamma", 250, "high", 0.3),     # Smaller, concentrated, higher risk
]


def generate_network(
    cities: Optional[List[CityConfig]] = None,
    n_high_risk_groups: int = 4,
    group_size_range: Tuple[int, int] = (10, 20),
    n_bridges: int = 15,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic person and transaction data.

    Parameters
    ----------
    cities : list of CityConfig, optional
        City configurations. Defaults to 3 cities with 1000 total persons.
    n_high_risk_groups : int
        Number of high-risk clusters to embed
    group_size_range : tuple
        Min and max size for high-risk groups
    n_bridges : int
        Number of "bridge" individuals who connect across cities
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    persons : pd.DataFrame
        Individual features including city, risk archetype, and base PD
    transactions : pd.DataFrame
        Transaction records between individuals
    """
    if seed is not None:
        np.random.seed(seed)

    if cities is None:
        cities = DEFAULT_CITIES

    # Generate persons
    persons = _generate_persons(cities)

    # Mark risk archetypes
    persons = _assign_risk_archetypes(persons, cities)

    # Create high-risk groups
    persons = _create_high_risk_groups(
        persons, n_high_risk_groups, group_size_range
    )

    # Mark bridge individuals
    persons = _mark_bridges(persons, n_bridges)

    # Compute base PD from features
    persons = _compute_base_pd(persons)

    # Generate transactions
    transactions = _generate_transactions(persons, cities)

    return persons, transactions


def _generate_persons(cities: List[CityConfig]) -> pd.DataFrame:
    """Generate person features for all cities."""
    all_persons = []
    person_id = 0

    for city_id, city in enumerate(cities):
        n = city.population

        # Age: 22-70, distribution varies by city
        if city.base_risk_level == "low":
            age = np.clip(np.random.normal(42, 12, n), 22, 70).astype(int)
        elif city.base_risk_level == "medium":
            age = np.clip(np.random.normal(38, 14, n), 22, 70).astype(int)
        else:
            age = np.clip(np.random.normal(35, 15, n), 22, 70).astype(int)

        # Income: correlated with city economic strength
        base_income = np.random.lognormal(10.5, 0.5, n)
        income = base_income * (0.6 + 0.8 * city.economic_strength)

        # Employment years
        max_emp = np.maximum(age - 22, 1)
        employment_years = np.random.beta(3, 2, n) * max_emp

        # Debt to income: higher in stressed cities
        dti_mean = 0.25 + 0.15 * (1 - city.economic_strength)
        debt_to_income = np.clip(np.random.beta(2, 5, n) + dti_mean, 0.1, 0.9)

        # Credit features
        num_credit_lines = np.random.poisson(3, n)
        missed_payments = np.random.poisson(0.5 + 1.0 * (1 - city.economic_strength), n)
        credit_utilization = np.clip(np.random.beta(2, 4, n), 0.05, 0.95)

        # Account age (months)
        account_age = np.random.gamma(4, 15, n)

        city_persons = pd.DataFrame({
            'person_id': range(person_id, person_id + n),
            'city_id': city_id,
            'city_name': city.name,
            'age': age,
            'income': np.round(income, 2),
            'employment_years': np.round(employment_years, 1),
            'debt_to_income': np.round(debt_to_income, 3),
            'num_credit_lines': num_credit_lines,
            'missed_payments': np.clip(missed_payments, 0, 10),
            'credit_utilization': np.round(credit_utilization, 3),
            'account_age_months': np.round(account_age, 0).astype(int),
        })

        all_persons.append(city_persons)
        person_id += n

    return pd.concat(all_persons, ignore_index=True)


def _assign_risk_archetypes(
    persons: pd.DataFrame,
    cities: List[CityConfig]
) -> pd.DataFrame:
    """
    Assign risk archetypes based on features.

    Archetypes:
    - 'low': ~70% of population, stable individuals
    - 'medium': ~20% of population, some stress signals
    - 'high': ~10% of population, clear distress
    """
    persons = persons.copy()
    n = len(persons)

    # Create risk score from features
    # Higher score = higher risk
    risk_score = (
        2.0 * persons['debt_to_income'] +
        0.5 * persons['missed_payments'] +
        1.5 * persons['credit_utilization'] -
        0.00001 * persons['income'] -
        0.02 * persons['employment_years'] -
        0.01 * persons['account_age_months']
    )

    # Add city effect
    city_risk_boost = persons['city_id'].map({
        0: 0.0,   # Alpha - low risk city
        1: 0.3,   # Beta - medium risk city
        2: 0.6,   # Gamma - high risk city
    })
    risk_score = risk_score + city_risk_boost

    # Assign archetypes based on percentiles
    percentiles = risk_score.rank(pct=True)

    persons['risk_archetype'] = 'low'
    persons.loc[percentiles > 0.70, 'risk_archetype'] = 'medium'
    persons.loc[percentiles > 0.90, 'risk_archetype'] = 'high'
    persons['individual_risk_score'] = np.round(risk_score, 4)

    # Initialize group membership
    persons['high_risk_group_id'] = -1
    persons['is_bridge'] = False

    return persons


def _create_high_risk_groups(
    persons: pd.DataFrame,
    n_groups: int,
    size_range: Tuple[int, int]
) -> pd.DataFrame:
    """
    Create high-risk groups (clusters of connected risky individuals).

    These represent:
    - Business networks under stress
    - Geographic clusters (same neighborhood)
    - Supply chain clusters
    """
    persons = persons.copy()

    for group_id in range(n_groups):
        group_size = np.random.randint(size_range[0], size_range[1] + 1)

        # Select a city to base the group in (bias toward riskier cities)
        city_weights = [0.2, 0.3, 0.5]  # Gamma gets more groups
        base_city = np.random.choice([0, 1, 2], p=city_weights)

        # Find candidates in this city who are medium or high risk
        candidates = persons[
            (persons['city_id'] == base_city) &
            (persons['risk_archetype'].isin(['medium', 'high'])) &
            (persons['high_risk_group_id'] == -1)
        ]['person_id'].values

        if len(candidates) < group_size:
            # Also include some low-risk individuals from same city
            low_risk = persons[
                (persons['city_id'] == base_city) &
                (persons['risk_archetype'] == 'low') &
                (persons['high_risk_group_id'] == -1)
            ]['person_id'].values
            candidates = np.concatenate([candidates, low_risk])

        if len(candidates) >= group_size:
            group_members = np.random.choice(
                candidates, size=group_size, replace=False
            )
            persons.loc[
                persons['person_id'].isin(group_members),
                'high_risk_group_id'
            ] = group_id

            # Boost risk for group members (group contagion effect)
            persons.loc[
                persons['person_id'].isin(group_members),
                'individual_risk_score'
            ] += 0.5

    return persons


def _mark_bridges(persons: pd.DataFrame, n_bridges: int) -> pd.DataFrame:
    """
    Mark bridge individuals who connect across cities.

    Bridges are:
    - Relatively low individual risk
    - But highly connected
    - Important for contagion propagation
    """
    persons = persons.copy()

    # Select low-risk individuals as bridges
    candidates = persons[
        (persons['risk_archetype'] == 'low') &
        (persons['high_risk_group_id'] == -1)
    ]['person_id'].values

    if len(candidates) >= n_bridges:
        bridges = np.random.choice(candidates, size=n_bridges, replace=False)
        persons.loc[persons['person_id'].isin(bridges), 'is_bridge'] = True

    return persons


def _compute_base_pd(persons: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base probability of default from features.

    This is the MARGINAL PD before considering network effects.
    """
    persons = persons.copy()

    # Logistic model coefficients
    z = (
        -3.0  # Intercept
        + 3.0 * (persons['debt_to_income'] - 0.3)
        - 0.00002 * (persons['income'] - 60000)
        + 0.3 * persons['missed_payments']
        + 2.0 * (persons['credit_utilization'] - 0.3)
        - 0.03 * persons['employment_years']
        - 0.008 * persons['account_age_months']
    )

    # Add archetype effect
    archetype_boost = persons['risk_archetype'].map({
        'low': -0.5,
        'medium': 0.3,
        'high': 1.0
    })
    z = z + archetype_boost

    # Add group membership effect
    in_group = (persons['high_risk_group_id'] >= 0).astype(float)
    z = z + 0.5 * in_group

    # Convert to probability
    base_pd = 1 / (1 + np.exp(-z))

    # Clip to reasonable range
    persons['base_pd'] = np.clip(np.round(base_pd, 4), 0.005, 0.60)

    # Generate a binary default label from Bernoulli(base_pd) for supervised training
    persons['default'] = np.random.binomial(1, persons['base_pd'].values).astype(int)

    return persons


def _generate_transactions(
    persons: pd.DataFrame,
    cities: List[CityConfig]
) -> pd.DataFrame:
    """
    Generate transaction network.

    Patterns:
    - Most transactions within same city (80%)
    - Bridge individuals have cross-city transactions
    - High-risk groups have dense internal transactions
    - Transaction amounts depend on income
    """
    n_persons = len(persons)
    transactions = []

    # Target: average 8 transactions per person
    n_transactions = int(n_persons * 8)

    # Pre-compute lookup tables
    city_members = {
        city_id: persons[persons['city_id'] == city_id]['person_id'].values
        for city_id in range(len(cities))
    }

    group_members = {
        gid: persons[persons['high_risk_group_id'] == gid]['person_id'].values
        for gid in persons['high_risk_group_id'].unique() if gid >= 0
    }

    bridge_ids = set(persons[persons['is_bridge']]['person_id'].values)

    for tx_id in range(n_transactions):
        # Select sender (weighted by income - more active)
        income_weights = persons['income'].values
        income_weights = income_weights / income_weights.sum()
        sender_id = np.random.choice(n_persons, p=income_weights)
        sender = persons.iloc[sender_id]

        # Determine receiver based on sender characteristics
        receiver_id = _select_receiver(
            sender, persons, city_members, group_members, bridge_ids
        )

        if receiver_id == sender_id:
            continue

        # Transaction amount based on sender income
        avg_amount = sender['income'] * 0.02
        amount = np.random.lognormal(np.log(avg_amount), 0.8)
        amount = np.clip(amount, 50, 50000)

        transactions.append({
            'transaction_id': tx_id,
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'amount': round(amount, 2),
            'is_within_city': sender['city_id'] == persons.iloc[receiver_id]['city_id'],
            'is_within_group': (
                sender['high_risk_group_id'] >= 0 and
                sender['high_risk_group_id'] == persons.iloc[receiver_id]['high_risk_group_id']
            )
        })

    return pd.DataFrame(transactions)


def _select_receiver(
    sender: pd.Series,
    persons: pd.DataFrame,
    city_members: dict,
    group_members: dict,
    bridge_ids: set
) -> int:
    """Select transaction receiver based on sender profile."""
    sender_id = sender['person_id']
    sender_city = sender['city_id']
    sender_group = sender['high_risk_group_id']

    # High-risk group members transact heavily within group
    if sender_group >= 0 and np.random.random() < 0.6:
        group = group_members[sender_group]
        candidates = [p for p in group if p != sender_id]
        if candidates:
            return np.random.choice(candidates)

    # Bridges transact across cities
    if sender_id in bridge_ids and np.random.random() < 0.5:
        other_cities = [c for c in city_members.keys() if c != sender_city]
        target_city = np.random.choice(other_cities)
        candidates = city_members[target_city]
        return np.random.choice(candidates)

    # Most transactions within same city
    if np.random.random() < 0.8:
        candidates = [p for p in city_members[sender_city] if p != sender_id]
        if candidates:
            return np.random.choice(candidates)

    # Cross-city transaction
    all_others = [p for p in range(len(persons)) if p != sender_id]
    return np.random.choice(all_others)


def get_summary_stats(persons: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """Get summary statistics for the generated data."""
    group_member_mask = persons['high_risk_group_id'] >= 0
    n_high_risk_groups = persons.loc[group_member_mask, 'high_risk_group_id'].nunique()

    return {
        'n_persons': len(persons),
        'n_transactions': len(transactions),
        'n_cities': persons['city_id'].nunique(),
        'persons_per_city': persons.groupby('city_name').size().to_dict(),
        'risk_archetype_counts': persons['risk_archetype'].value_counts().to_dict(),
        'n_high_risk_groups': int(n_high_risk_groups),
        'n_high_risk_group_members': int(group_member_mask.sum()),
        'n_bridges': persons['is_bridge'].sum(),
        'avg_base_pd': persons['base_pd'].mean(),
        'pd_by_archetype': persons.groupby('risk_archetype')['base_pd'].mean().to_dict(),
        'pd_by_city': persons.groupby('city_name')['base_pd'].mean().to_dict(),
        'within_city_tx_pct': transactions['is_within_city'].mean(),
        'within_group_tx_pct': transactions['is_within_group'].mean(),
    }


def generate_fraud_rings(
    persons: pd.DataFrame,
    transactions: pd.DataFrame,
    n_rings: int = 3,
    ring_size_range: Tuple[int, int] = (4, 8),
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add fraud ring structures to existing network data.

    Fraud rings are characterized by:
    - Tightly connected clusters with circular money flows
    - Higher than average transaction volumes within the ring
    - Members have elevated PDs due to coordinated risky behavior
    - Unusual transaction patterns (regular amounts, timing)

    Parameters
    ----------
    persons : pd.DataFrame
        Existing person data
    transactions : pd.DataFrame
        Existing transaction data
    n_rings : int
        Number of fraud rings to create
    ring_size_range : tuple
        Min and max members per ring
    seed : int, optional
        Random seed

    Returns
    -------
    persons : pd.DataFrame
        Updated with fraud ring membership
    transactions : pd.DataFrame
        Updated with fraud ring transactions
    """
    if seed is not None:
        np.random.seed(seed)

    persons = persons.copy()
    transactions = transactions.copy()

    # Initialize fraud ring column
    persons['is_fraud_ring'] = False
    persons['fraud_ring_id'] = -1

    # Track new transactions
    new_transactions = []
    next_tx_id = transactions['transaction_id'].max() + 1

    for ring_id in range(n_rings):
        ring_size = np.random.randint(ring_size_range[0], ring_size_range[1] + 1)

        # Select candidates - prefer medium risk individuals (not too obvious)
        candidates = persons[
            (persons['risk_archetype'].isin(['medium', 'high'])) &
            (~persons['is_fraud_ring']) &
            (persons['high_risk_group_id'] == -1)  # Not already in a group
        ]['person_id'].values

        if len(candidates) < ring_size:
            # Fall back to any non-fraud-ring members
            candidates = persons[~persons['is_fraud_ring']]['person_id'].values

        if len(candidates) < ring_size:
            continue

        # Select ring members
        ring_members = np.random.choice(candidates, size=ring_size, replace=False)

        # Mark as fraud ring members
        persons.loc[persons['person_id'].isin(ring_members), 'is_fraud_ring'] = True
        persons.loc[persons['person_id'].isin(ring_members), 'fraud_ring_id'] = ring_id

        # Boost PD for fraud ring members (they're engaged in risky behavior)
        persons.loc[persons['person_id'].isin(ring_members), 'base_pd'] = np.clip(
            persons.loc[persons['person_id'].isin(ring_members), 'base_pd'] * 1.5,
            0, 0.8
        )

        # Create circular transaction flow (A -> B -> C -> ... -> A)
        # This is a key fraud pattern
        base_amount = np.random.uniform(1000, 5000)

        for i in range(ring_size):
            sender = ring_members[i]
            receiver = ring_members[(i + 1) % ring_size]

            # Multiple circular transactions with similar amounts
            for _ in range(np.random.randint(3, 8)):
                amount = base_amount * np.random.uniform(0.9, 1.1)  # Small variation

                new_transactions.append({
                    'transaction_id': next_tx_id,
                    'sender_id': sender,
                    'receiver_id': receiver,
                    'amount': round(amount, 2),
                    'is_within_city': persons.loc[
                        persons['person_id'] == sender, 'city_id'
                    ].values[0] == persons.loc[
                        persons['person_id'] == receiver, 'city_id'
                    ].values[0],
                    'is_within_group': True,
                    'is_fraud_ring_tx': True
                })
                next_tx_id += 1

        # Add some cross-ring transactions (not just circular)
        n_cross = np.random.randint(5, 15)
        for _ in range(n_cross):
            sender = np.random.choice(ring_members)
            receiver = np.random.choice([m for m in ring_members if m != sender])
            amount = base_amount * np.random.uniform(0.5, 1.5)

            new_transactions.append({
                'transaction_id': next_tx_id,
                'sender_id': sender,
                'receiver_id': receiver,
                'amount': round(amount, 2),
                'is_within_city': persons.loc[
                    persons['person_id'] == sender, 'city_id'
                ].values[0] == persons.loc[
                    persons['person_id'] == receiver, 'city_id'
                ].values[0],
                'is_within_group': True,
                'is_fraud_ring_tx': True
            })
            next_tx_id += 1

    # Add is_fraud_ring_tx column to original transactions
    transactions['is_fraud_ring_tx'] = False

    # Combine transactions
    if new_transactions:
        new_tx_df = pd.DataFrame(new_transactions)
        transactions = pd.concat([transactions, new_tx_df], ignore_index=True)

    return persons, transactions


# Alias for backward compatibility with main.py
generate_synthetic_network = generate_network


if __name__ == '__main__':
    print("Generating network with 1000 persons across 3 cities...")
    persons, transactions = generate_network(seed=42)

    stats = get_summary_stats(persons, transactions)

    print("\n=== NETWORK SUMMARY ===")
    print(f"Total persons: {stats['n_persons']}")
    print(f"Total transactions: {stats['n_transactions']}")

    print("\n--- By City ---")
    for city, count in stats['persons_per_city'].items():
        pd_city = stats['pd_by_city'][city]
        print(f"  {city}: {count} persons, avg PD = {pd_city:.2%}")

    print("\n--- Risk Archetypes ---")
    for archetype, count in stats['risk_archetype_counts'].items():
        pd_arch = stats['pd_by_archetype'][archetype]
        print(f"  {archetype}: {count} persons, avg PD = {pd_arch:.2%}")

    print(f"\n--- Special Groups ---")
    print(f"  High-risk groups: {stats['n_high_risk_groups']}")
    print(f"  High-risk group members: {stats['n_high_risk_group_members']}")
    print(f"  Bridge individuals: {stats['n_bridges']}")

    print(f"\n--- Transaction Patterns ---")
    print(f"  Within-city transactions: {stats['within_city_tx_pct']:.1%}")
    print(f"  Within-group transactions: {stats['within_group_tx_pct']:.1%}")

    print("\n--- Sample Persons ---")
    print(persons[['person_id', 'city_name', 'risk_archetype', 'base_pd',
                   'high_risk_group_id', 'is_bridge']].head(20).to_string())
