"""
TV Advertising Data Generator
=============================
Generates realistic simulated TV advertising datasets for a DTC brand
running a national TV campaign across 8 DMAs over 12 months.

Brand Profile:
- Consumer brand (mattress/meal-kit category)
- $15M annual budget across TV + digital
- 8 DMAs: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, Dallas, Atlanta
- Campaign: 2023-01-01 to 2023-12-31
- Networks: ABC, CBS, NBC, FOX + ESPN, CNN, HGTV, Food Network, TNT, TBS

Outputs:
- data/ad_airings.csv         (~3 MB, ~25K rows)
- data/web_traffic.csv        (~100+ MB, ~700K rows — for Colab only)
- data/weekly_spend.csv       (~5 KB, 52 rows)
- data/hourly_traffic_agg.csv (~10 MB — for Streamlit)
- data/daily_traffic.csv      (~2 MB — for Streamlit)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import time

np.random.seed(42)

# ============================================
# CONSTANTS
# ============================================

DMAS = ['New_York', 'Los_Angeles', 'Chicago', 'Houston',
        'Phoenix', 'Philadelphia', 'Dallas', 'Atlanta']

DMA_WEIGHTS = {
    'New_York': 0.22, 'Los_Angeles': 0.16, 'Chicago': 0.10,
    'Houston': 0.08, 'Phoenix': 0.06, 'Philadelphia': 0.09,
    'Dallas': 0.08, 'Atlanta': 0.07
}

DMA_BASE_TRAFFIC = {
    'New_York': 5000, 'Los_Angeles': 3800, 'Chicago': 2200,
    'Houston': 1500, 'Phoenix': 1200, 'Philadelphia': 1800,
    'Dallas': 1400, 'Atlanta': 1300
}

NETWORKS = {
    'broadcast': ['ABC', 'CBS', 'NBC', 'FOX'],
    'cable': ['ESPN', 'CNN', 'HGTV', 'Food_Network', 'TNT', 'TBS']
}

DAYPARTS = {
    'early_morning': (5, 9),
    'daytime': (9, 16),
    'early_fringe': (16, 18),
    'primetime': (20, 23),
    'late_night': (23, 2),
    'sports': (12, 22),
}

DAYPART_PROBS = [0.05, 0.15, 0.10, 0.40, 0.10, 0.20]

CREATIVES = ['creative_A_brand', 'creative_B_product',
             'creative_C_testimonial', 'creative_D_promo']

SEASONAL_MULTIPLIERS = {
    1: 1.3, 2: 1.1, 3: 0.9, 4: 0.8, 5: 0.8, 6: 0.7,
    7: 0.7, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.5, 12: 1.4
}

# US holidays in 2023 (affects traffic patterns)
HOLIDAYS_2023 = {
    datetime(2023, 1, 1), datetime(2023, 1, 2),   # New Year
    datetime(2023, 1, 16),                          # MLK Day
    datetime(2023, 2, 12),                          # Super Bowl Sunday
    datetime(2023, 2, 20),                          # Presidents Day
    datetime(2023, 5, 29),                          # Memorial Day
    datetime(2023, 7, 4),                           # July 4th
    datetime(2023, 9, 4),                           # Labor Day
    datetime(2023, 10, 9),                          # Columbus Day
    datetime(2023, 11, 23), datetime(2023, 11, 24), # Thanksgiving
    datetime(2023, 12, 25), datetime(2023, 12, 26), # Christmas
    datetime(2023, 12, 31),                          # NYE
}


# ============================================
# 1. AD AIRINGS TABLE (~25,000 rows)
# ============================================

def generate_ad_airings():
    """
    Generate TV ad airing records with realistic patterns:
    - Heavier spend in Q4 (holiday) and Q1 (New Year)
    - Primetime (8-11pm) costs 3-5x more than daytime
    - Sports programming has highest impressions
    - 15s, 30s, 60s ad units with different costs
    - Multiple creative versions (4 creatives, rotated)
    - National airings affect all DMAs; local airings affect one DMA

    CPM ranges (cost per thousand impressions):
    - Broadcast primetime: $30-50
    - Broadcast daytime: $8-15
    - Cable primetime: $12-25
    - Cable daytime: $5-12
    - Sports events: $40-80
    - Late night: $5-10
    """
    airings = []
    airing_id = 0

    for day_offset in range(365):
        date = datetime(2023, 1, 1) + timedelta(days=day_offset)
        month = date.month
        dow = date.weekday()

        seasonal = SEASONAL_MULTIPLIERS[month]

        # A $15M budget brand runs ~8-12 unique airings/day nationally
        # (each expands to 8 DMA rows if national -> ~25K total rows)
        weekend_factor = 0.85 if dow >= 5 else 1.0
        base_airings = int(np.random.poisson(10 * seasonal * weekend_factor))

        # Super Bowl Sunday special: extra airings on Feb 12
        if date == datetime(2023, 2, 12):
            base_airings = int(base_airings * 2)

        for _ in range(base_airings):
            # Pick network type — broadcast slightly more likely on weekdays
            broadcast_prob = 0.50 if dow < 5 else 0.40
            is_broadcast = np.random.random() < broadcast_prob
            if is_broadcast:
                network = np.random.choice(NETWORKS['broadcast'])
                net_type = 'broadcast'
            else:
                network = np.random.choice(NETWORKS['cable'])
                net_type = 'cable'

            # ESPN more likely on weekends
            if dow >= 5 and net_type == 'cable' and np.random.random() < 0.3:
                network = 'ESPN'

            # Pick daypart
            daypart = np.random.choice(list(DAYPARTS.keys()), p=DAYPART_PROBS)
            hour_start, hour_end = DAYPARTS[daypart]
            if hour_end < hour_start:
                hour = np.random.choice(list(range(hour_start, 24)) + list(range(0, hour_end)))
            else:
                hour = np.random.randint(hour_start, max(hour_start + 1, hour_end))
            minute = np.random.choice([0, 15, 30, 45])

            # Ad duration
            duration = np.random.choice([15, 30, 60], p=[0.25, 0.60, 0.15])

            # National vs local
            is_national = np.random.random() < 0.70
            affected_dmas = DMAS if is_national else [np.random.choice(DMAS)]

            # Creative rotation — newer creatives more likely later in year
            if month <= 3:
                creative_weights = [0.50, 0.30, 0.15, 0.05]
            elif month <= 6:
                creative_weights = [0.30, 0.35, 0.25, 0.10]
            elif month <= 9:
                creative_weights = [0.10, 0.25, 0.35, 0.30]
            else:
                creative_weights = [0.05, 0.15, 0.30, 0.50]
            creative = np.random.choice(CREATIVES, p=creative_weights)

            # Program name generation for realism
            program_map = {
                ('ABC', 'primetime'): ['Grey\'s Anatomy', 'The Bachelor', 'Station 19'],
                ('CBS', 'primetime'): ['NCIS', 'Survivor', 'Blue Bloods'],
                ('NBC', 'primetime'): ['Law & Order', 'Chicago Fire', 'The Voice'],
                ('FOX', 'primetime'): ['9-1-1', 'The Masked Singer', 'Next Level Chef'],
                ('ESPN', 'sports'): ['Monday Night Football', 'SportsCenter', 'NBA Countdown'],
                ('CNN', 'primetime'): ['Anderson Cooper 360', 'CNN Tonight'],
                ('HGTV', 'primetime'): ['Fixer Upper', 'House Hunters'],
                ('Food_Network', 'primetime'): ['Chopped', 'Diners Drive-Ins'],
                ('TNT', 'primetime'): ['NBA Basketball', 'AEW Dynamite'],
                ('TBS', 'primetime'): ['MLB Postseason', 'American Dad'],
            }
            programs = program_map.get((network, daypart),
                       program_map.get((network, 'primetime'), ['General Programming']))
            program = np.random.choice(programs)

            # CPM based on network type, daypart, and special events
            cpm_map = {
                ('broadcast', 'primetime'): (30, 50),
                ('broadcast', 'daytime'): (8, 15),
                ('broadcast', 'early_morning'): (6, 12),
                ('broadcast', 'early_fringe'): (12, 22),
                ('broadcast', 'sports'): (40, 80),
                ('broadcast', 'late_night'): (5, 10),
                ('cable', 'primetime'): (12, 25),
                ('cable', 'daytime'): (5, 12),
                ('cable', 'early_morning'): (3, 8),
                ('cable', 'early_fringe'): (8, 16),
                ('cable', 'sports'): (25, 55),
                ('cable', 'late_night'): (4, 9),
            }
            cpm_low, cpm_high = cpm_map.get((net_type, daypart), (5, 15))
            base_cpm = np.random.uniform(cpm_low, cpm_high)

            # Super Bowl premium
            if date == datetime(2023, 2, 12) and daypart == 'sports':
                base_cpm *= 5.0

            # Duration cost multiplier
            dur_mult = {15: 0.5, 30: 1.0, 60: 1.8}[duration]

            # Total impressions for the airing (across all DMAs)
            # A mid-size DTC brand with $8.25M TV budget buys ~100K-250K
            # national impressions per spot, yielding ~$1.5K-$4K per airing
            total_impressions_base = 55_000
            daypart_mult = {
                'primetime': 1.5, 'sports': 2.0,
                'daytime': 0.6, 'early_morning': 0.3,
                'early_fringe': 0.8, 'late_night': 0.4
            }[daypart]
            total_impressions = int(
                total_impressions_base * daypart_mult *
                np.random.uniform(0.7, 1.3) *
                (1.2 if dow >= 5 else 1.0)
            )

            # Total cost for the airing (split across DMAs)
            total_cost = (total_impressions / 1000) * base_cpm * dur_mult
            n_dmas = len(affected_dmas)

            for dma in affected_dmas:
                # DMA gets its proportional share of impressions
                dma_weight = DMA_WEIGHTS.get(dma, 0.05)
                total_weight = sum(DMA_WEIGHTS.get(d, 0.05) for d in affected_dmas)
                dma_share = dma_weight / total_weight

                impressions = int(total_impressions * dma_share * np.random.uniform(0.9, 1.1))
                cost = round(total_cost * dma_share, 2)

                # Reach and frequency estimates
                reach_pct = min(0.95, daypart_mult * 0.15 * np.random.uniform(0.8, 1.2))
                frequency = impressions / max(1, reach_pct * dma_weight * 50_000_000)

                airings.append({
                    'airing_id': f'A{airing_id:06d}',
                    'date': date.strftime('%Y-%m-%d'),
                    'timestamp': f"{date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}:00",
                    'network': network,
                    'network_type': net_type,
                    'daypart': daypart,
                    'program': program,
                    'dma': dma,
                    'duration_seconds': duration,
                    'creative_id': creative,
                    'impressions': impressions,
                    'reach_pct': round(reach_pct, 4),
                    'frequency': round(frequency, 2),
                    'cost': round(cost, 2),
                    'cpm': round(base_cpm * dur_mult, 2),
                    'is_national': is_national,
                    'is_holiday': date in HOLIDAYS_2023,
                })
                airing_id += 1

    return pd.DataFrame(airings)


# ============================================
# 2. WEB TRAFFIC TABLE (~700,000 rows)
# ============================================

def generate_web_traffic(airings_df):
    """
    Generate 5-minute-bucket web traffic per DMA with TV ad response patterns.

    Realism:
    - Baseline follows hour-of-day curve + day-of-week + seasonality
    - TV response: sharp spike 2-5 min after airing, decay over 15 min
    - Response magnitude depends on daypart, creative, impressions
    - Not every airing produces detectable lift (noise)
    - Organic growth trend: +15% over the year
    """
    print("  Generating baseline traffic grid...")
    start = time.time()

    records = []

    for day_offset in range(365):
        date = datetime(2023, 1, 1) + timedelta(days=day_offset)
        month = date.month
        dow = date.weekday()

        organic_trend = 1.0 + (day_offset / 365) * 0.15
        dow_mult = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.05, 4: 1.1,
                    5: 0.85, 6: 0.80}[dow]
        seasonal = {1: 1.1, 2: 0.95, 3: 0.9, 4: 0.85, 5: 0.85, 6: 0.80,
                    7: 0.80, 8: 0.85, 9: 0.95, 10: 1.1, 11: 1.3, 12: 1.2}[month]

        is_holiday = date in HOLIDAYS_2023
        holiday_mult = 0.75 if is_holiday else 1.0  # less work-traffic on holidays

        date_str = date.strftime('%Y-%m-%d')

        for dma in DMAS:
            dma_base = DMA_BASE_TRAFFIC[dma]

            for hour in range(24):
                # Hour-of-day curve: peaks at 10am (work browsing) and 8pm (evening)
                hour_curve = (
                    0.2 +
                    0.5 * np.exp(-((hour - 10) ** 2) / 8) +
                    0.8 * np.exp(-((hour - 20) ** 2) / 6)
                )

                for minute_bucket in range(0, 60, 5):
                    # Base sessions for this 5-min bucket
                    base = ((dma_base / 288) * hour_curve * dow_mult *
                            seasonal * organic_trend * holiday_mult)

                    sessions = max(0, int(np.random.poisson(max(1, base))))

                    # Source split varies by hour
                    is_work_hours = 9 <= hour <= 17
                    direct_pct = 0.30 if is_work_hours else 0.40
                    organic_pct = 0.35 if is_work_hours else 0.25
                    paid_pct = 0.22 if is_work_hours else 0.18
                    social_pct = 1.0 - direct_pct - organic_pct - paid_pct

                    direct = int(sessions * direct_pct)
                    organic = int(sessions * organic_pct)
                    paid = int(sessions * paid_pct)
                    social = sessions - direct - organic - paid  # remainder

                    # Conversion rate varies by hour (higher during work, lower late night)
                    base_conv_rate = 0.028 if is_work_hours else 0.022
                    conversions = int(np.random.binomial(max(0, sessions), base_conv_rate))

                    # Revenue per conversion: $80-200, higher for direct traffic
                    avg_rev = np.random.uniform(90, 180)
                    revenue = round(conversions * avg_rev, 2)

                    mobile_pct = round(0.55 + 0.10 * (hour >= 18) - 0.08 * is_work_hours, 2)

                    ts = f"{date_str} {hour:02d}:{minute_bucket:02d}:00"

                    records.append({
                        'date': date_str,
                        'timestamp': ts,
                        'dma': dma,
                        'hour': hour,
                        'sessions': sessions,
                        'direct_sessions': direct,
                        'organic_sessions': organic,
                        'paid_sessions': paid,
                        'social_sessions': max(0, social),
                        'conversions': conversions,
                        'revenue': revenue,
                        'mobile_pct': mobile_pct,
                        'is_holiday': is_holiday,
                    })

    traffic_df = pd.DataFrame(records)
    elapsed = time.time() - start
    print(f"  Baseline grid generated in {elapsed:.1f}s: {len(traffic_df)} rows")

    # ---- ADD TV AD RESPONSE SPIKES ----
    print("  Adding TV response spikes...")
    start = time.time()

    # Build a lookup index for fast matching
    traffic_df['_idx'] = traffic_df.index
    ts_dma_index = traffic_df.set_index(['timestamp', 'dma'])['_idx']

    creative_effectiveness = {
        'creative_A_brand': 0.8,
        'creative_B_product': 1.2,
        'creative_C_testimonial': 1.0,
        'creative_D_promo': 1.5,
    }

    # Vectorized response generation
    n_airings = len(airings_df)
    response_mask = np.random.random(n_airings) < 0.60  # 60% create response
    responding = airings_df[response_mask].copy()

    sessions_add = np.zeros(len(traffic_df), dtype=int)
    direct_add = np.zeros(len(traffic_df), dtype=int)
    organic_add = np.zeros(len(traffic_df), dtype=int)
    conv_add = np.zeros(len(traffic_df), dtype=int)
    rev_add = np.zeros(len(traffic_df), dtype=float)

    for _, airing in responding.iterrows():
        air_time = pd.to_datetime(airing['timestamp'])
        dma = airing['dma']
        impressions = airing['impressions']

        eff = creative_effectiveness[airing['creative_id']]
        # Visit rate: ~0.05% of impressions result in a website visit
        # For a DTC brand, this yields 3-15 sessions per airing per DMA
        visit_rate = 0.0005 * eff * np.random.uniform(0.5, 1.5)
        base_response = impressions * visit_rate

        for delay_min in range(0, 20, 5):
            if delay_min < 5:
                response_mult = 1.0
            elif delay_min < 10:
                response_mult = 0.5
            else:
                response_mult = 0.15

            response_sessions = max(0, int(
                base_response * response_mult * np.random.uniform(0.7, 1.3)
            ))
            if response_sessions == 0:
                continue

            target_time = air_time + timedelta(minutes=delay_min)
            target_minute = (target_time.minute // 5) * 5
            target_ts = target_time.strftime(f'%Y-%m-%d %H:{target_minute:02d}:00')

            key = (target_ts, dma)
            if key in ts_dma_index.index:
                idx = ts_dma_index.loc[key]
                if isinstance(idx, pd.Series):
                    idx = idx.iloc[0]
                sessions_add[idx] += response_sessions
                direct_add[idx] += int(response_sessions * 0.70)
                organic_add[idx] += int(response_sessions * 0.30)
                resp_conv = int(np.random.binomial(response_sessions, 0.04))
                conv_add[idx] += resp_conv
                rev_add[idx] += resp_conv * np.random.uniform(95, 190)

    traffic_df['sessions'] += sessions_add
    traffic_df['direct_sessions'] += direct_add
    traffic_df['organic_sessions'] += organic_add
    traffic_df['conversions'] += conv_add
    traffic_df['revenue'] += np.round(rev_add, 2)

    traffic_df.drop(columns=['_idx'], inplace=True)

    elapsed = time.time() - start
    print(f"  Response spikes added in {elapsed:.1f}s")

    return traffic_df


# ============================================
# 3. WEEKLY CHANNEL SPEND TABLE (52 rows)
# ============================================

def _adstock(series, decay):
    """Simple geometric adstock for data generation."""
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for t in range(1, len(series)):
        result[t] = series[t] + decay * result[t - 1]
    return result


def _hill(x, alpha, K):
    """Hill saturation for data generation."""
    x = np.asarray(x, dtype=float)
    return np.power(x, alpha) / (np.power(x, alpha) + np.power(K, alpha))


def generate_weekly_spend():
    """
    Weekly advertising spend across all channels.
    Annual budget: ~$15M
    Split: TV 55% (broadcast 30%, cable 20%, streaming 5%),
           Paid Search 20%, Social 15%, Display 10%

    Revenue is generated WITH a causal relationship to spend:
    revenue = base + sum(beta_i * saturated(adstocked(spend_i))) + seasonal + noise

    This ensures the MMM can recover meaningful coefficients.
    """
    start_date = datetime(2023, 1, 1)

    # First pass: generate all weekly spends
    raw_weeks = []
    for week in range(1, 53):
        week_start = start_date + timedelta(weeks=week - 1)
        month = min(12, week_start.month)
        seasonal = SEASONAL_MULTIPLIERS[month]

        base_weekly = 15_000_000 / 52  # ~$288K/week

        tv_broadcast = base_weekly * 0.30 * seasonal * np.random.uniform(0.85, 1.15)
        tv_cable = base_weekly * 0.20 * seasonal * np.random.uniform(0.85, 1.15)
        tv_streaming = base_weekly * 0.05 * seasonal * np.random.uniform(0.70, 1.30)
        paid_search = base_weekly * 0.20 * np.random.uniform(0.90, 1.10)  # less seasonal
        social = base_weekly * 0.15 * seasonal * np.random.uniform(0.85, 1.15)
        display = base_weekly * 0.10 * np.random.uniform(0.85, 1.15)

        raw_weeks.append({
            'week': week,
            'week_start': week_start.strftime('%Y-%m-%d'),
            'month': month,
            'tv_broadcast_spend': tv_broadcast,
            'tv_cable_spend': tv_cable,
            'tv_streaming_spend': tv_streaming,
            'paid_search_spend': paid_search,
            'social_spend': social,
            'display_spend': display,
        })

    df = pd.DataFrame(raw_weeks)

    # True channel parameters (these are what the MMM should recover)
    # beta = max weekly revenue contribution when channel is fully saturated
    true_params = {
        'tv_broadcast_spend':  {'decay': 0.80, 'alpha': 0.65, 'K': 120000, 'beta': 160_000},
        'tv_cable_spend':      {'decay': 0.75, 'alpha': 0.60, 'K': 80000,  'beta': 110_000},
        'tv_streaming_spend':  {'decay': 0.50, 'alpha': 0.55, 'K': 25000,  'beta': 40_000},
        'paid_search_spend':   {'decay': 0.30, 'alpha': 0.70, 'K': 70000,  'beta': 200_000},
        'social_spend':        {'decay': 0.40, 'alpha': 0.60, 'K': 55000,  'beta': 90_000},
        'display_spend':       {'decay': 0.35, 'alpha': 0.50, 'K': 35000,  'beta': 35_000},
    }

    # Add flight patterns: some weeks certain channels are off/reduced
    # This creates variation independent of seasonality (critical for MMM)
    for col in ['tv_broadcast_spend', 'tv_cable_spend']:
        # TV flights: 2-week off periods every ~8 weeks
        for flight_start in [6, 14, 22, 30, 38, 46]:
            if np.random.random() < 0.5:  # 50% chance of each flight gap
                gap_weeks = np.random.randint(1, 3)
                for w in range(flight_start, min(flight_start + gap_weeks, 52)):
                    df.loc[w, col] *= np.random.uniform(0.05, 0.20)

    for col in ['social_spend', 'display_spend']:
        # Digital pauses: occasional test/hold periods
        for w in range(52):
            if np.random.random() < 0.08:  # 8% of weeks
                df.loc[w, col] *= np.random.uniform(0.10, 0.30)

    # Generate revenue as a function of transformed spend
    # revenue = base + sum(beta_i * hill(adstock(spend_i))) + seasonal + noise
    base_revenue = 400_000  # $400K base weekly revenue (organic/brand)
    month_vals = df['month'].values
    seasonal_effect = (np.sin(2 * np.pi * month_vals / 12) * 80_000 +
                       np.cos(2 * np.pi * month_vals / 12) * 40_000)
    trend_effect = np.arange(52) / 52 * 60_000  # +$60K growth over year

    channel_revenue = np.zeros(52)
    for col, params in true_params.items():
        spend = df[col].values
        adstocked = _adstock(spend, params['decay'])
        saturated = _hill(adstocked, params['alpha'], params['K'])
        channel_revenue += params['beta'] * saturated

    total_revenue = (base_revenue + seasonal_effect + trend_effect +
                     channel_revenue +
                     np.random.normal(0, 30_000, 52))  # modest noise

    total_revenue = np.maximum(total_revenue, 100_000)  # floor

    # Compute derived fields
    weeks = []
    for i, row in df.iterrows():
        total_spend = (row['tv_broadcast_spend'] + row['tv_cable_spend'] +
                       row['tv_streaming_spend'] + row['paid_search_spend'] +
                       row['social_spend'] + row['display_spend'])

        seasonal = SEASONAL_MULTIPLIERS[row['month']]

        # Brand search correlates with TV spend (brand awareness effect)
        tv_total = row['tv_broadcast_spend'] + row['tv_cable_spend'] + row['tv_streaming_spend']
        brand_base = 50000 * seasonal * (1 + tv_total / 200000)
        brand_search = int(np.random.normal(brand_base, brand_base * 0.10))

        total_sessions = int(total_revenue[i] / 5.5 + np.random.normal(0, 5000))
        total_conversions = int(total_sessions * np.random.uniform(0.028, 0.035))

        weeks.append({
            'week': row['week'],
            'week_start': row['week_start'],
            'month': row['month'],
            'tv_broadcast_spend': round(row['tv_broadcast_spend'], 2),
            'tv_cable_spend': round(row['tv_cable_spend'], 2),
            'tv_streaming_spend': round(row['tv_streaming_spend'], 2),
            'paid_search_spend': round(row['paid_search_spend'], 2),
            'social_spend': round(row['social_spend'], 2),
            'display_spend': round(row['display_spend'], 2),
            'total_spend': round(total_spend, 2),
            'total_sessions': max(0, total_sessions),
            'total_conversions': max(0, total_conversions),
            'total_revenue': round(float(total_revenue[i]), 2),
            'brand_search_volume': max(0, brand_search),
        })

    return pd.DataFrame(weeks)


# ============================================
# GENERATE AND SAVE
# ============================================

def main():
    os.makedirs('data', exist_ok=True)

    print("=" * 60)
    print("TV ADVERTISING DATA GENERATOR")
    print("=" * 60)

    print("\n[1/3] Generating ad airings...")
    airings = generate_ad_airings()
    print(f"  -> {len(airings):,} airing records")
    print(f"  -> Total cost: ${airings['cost'].sum():,.0f}")
    print(f"  -> Date range: {airings['date'].min()} to {airings['date'].max()}")
    print(f"  -> Networks: {airings['network'].nunique()}")
    print(f"  -> National: {airings['is_national'].mean():.1%}")

    print("\n[2/3] Generating web traffic (this may take a few minutes)...")
    traffic = generate_web_traffic(airings)
    print(f"  -> {len(traffic):,} traffic records")
    print(f"  -> Total sessions: {traffic['sessions'].sum():,}")
    print(f"  -> Total conversions: {traffic['conversions'].sum():,}")

    print("\n[3/3] Generating weekly channel spend...")
    weekly_spend = generate_weekly_spend()
    print(f"  -> {len(weekly_spend)} weeks")
    print(f"  -> Total annual spend: ${weekly_spend['total_spend'].sum():,.0f}")
    print(f"  -> Total annual revenue: ${weekly_spend['total_revenue'].sum():,.0f}")

    # Save full data
    print("\nSaving datasets...")
    airings.to_csv('data/ad_airings.csv', index=False)
    traffic.to_csv('data/web_traffic.csv', index=False)
    weekly_spend.to_csv('data/weekly_spend.csv', index=False)

    # Aggregated data for Streamlit (traffic is too large for repo)
    print("Creating aggregated views...")

    # Hourly per DMA
    hourly_traffic = traffic.copy()
    hourly_traffic['hour'] = pd.to_datetime(hourly_traffic['timestamp']).dt.hour
    hourly_agg = hourly_traffic.groupby(['date', 'dma', 'hour']).agg({
        'sessions': 'sum',
        'direct_sessions': 'sum',
        'organic_sessions': 'sum',
        'paid_sessions': 'sum',
        'social_sessions': 'sum',
        'conversions': 'sum',
        'revenue': 'sum',
    }).reset_index()
    hourly_agg.to_csv('data/hourly_traffic_agg.csv', index=False)

    # Daily per DMA
    daily_agg = traffic.groupby(['date', 'dma']).agg({
        'sessions': 'sum',
        'direct_sessions': 'sum',
        'organic_sessions': 'sum',
        'paid_sessions': 'sum',
        'social_sessions': 'sum',
        'conversions': 'sum',
        'revenue': 'sum',
    }).reset_index()
    daily_agg.to_csv('data/daily_traffic.csv', index=False)

    # Print file sizes
    print("\nFile sizes:")
    for f in ['ad_airings.csv', 'web_traffic.csv', 'weekly_spend.csv',
              'hourly_traffic_agg.csv', 'daily_traffic.csv']:
        path = f'data/{f}'
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  {f}: {size:.1f} MB")

    print("\nData generation complete!")
    return airings, traffic, weekly_spend


if __name__ == '__main__':
    main()
