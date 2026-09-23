# Sales Analysis Summary

**Period:** 2025-07-02 to 2025-09-01
**Total sales:** $2,667,260.23
**Units sold:** 12,281
**Valid orders:** 10,218 / 10,692
**Distinct SKUs:** 3,191

## Top Performing SKUs

| SKU | Item | Revenue | Units | Orders |
|---|---|---|---|---|
| MFK-46119 | MFK-46119 | $52,085.34 | 185 | 173 |
| IR150924 | IR150924 | $38,515.53 | 30 | 30 |
| MFK-87865 | MFK-87865 | $22,728.01 | 47 | 47 |
| MFK-107221 | MFK-107221 | $20,676.36 | 56 | 56 |
| MFK-363098 | MFK-363098 | $18,439.80 | 91 | 90 |
| MFK-62000 | MFK-62000 | $18,164.32 | 57 | 55 |
| MFK-991105 | MFK-991105 | $16,988.13 | 55 | 55 |
| MFK-394147 | MFK-394147 | $14,308.84 | 39 | 35 |
| MFK-625892 | MFK-625892 | $13,360.78 | 46 | 46 |
| IR412953 | IR412953 | $13,224.53 | 14 | 14 |

## Low / Non-Moving SKUs

| SKU | Item | Revenue | Units | Orders |
|---|---|---|---|---|
| IR982698 | IR982698 | $0.00 | 1 | 1 |
| KHT-LSH-1647-GLX | KHT-LSH-1647-GLX | $0.00 | 2 | 1 |
| IR795505 | IR795505 | $3.59 | 2 | 2 |
| IR151932 | IR151932 | $3.77 | 1 | 1 |
| IR696598 | IR696598 | $9.34 | 1 | 1 |
| IR888836 | IR888836 | $9.44 | 1 | 1 |
| IR104020 | IR104020 | $10.98 | 1 | 1 |
| MFK-504705 | MFK-504705 | $12.70 | 1 | 1 |
| IR628654 | IR628654 | $13.18 | 1 | 1 |
| UQN-NH92292 | UQN-NH92292 | $13.30 | 1 | 1 |

## Business Insights

- Total completed revenue: $2,667,260.23 across 12,281 units and 10,218 valid order lines.
- Top 10 SKUs account for 8.6% of total sales — heavy revenue concentration.
- Cancellation rate: 4.2% (452 orders).
- 14.9% of orders (1,593) have unmapped carrier — data gap, not fulfilment failure.
- Monthly revenue: Jul 2025: $1,219,866, Aug 2025: $1,395,310, Sep 2025: $52,084.

## Data Quality Notes

- Cancelled and Pending excluded from revenue/units (not completed transactions).
- OJ Shipping kept as valid (in fulfillment, not failed).
- Carrier Not Mapped (1593 rows) kept in revenue but flagged as data gap.
- 36 Processed orders have $0 price — flagged, not dropped.
- Vendor SKUs and sales SKUs share no common key (confirmed at load time).