# AI Engineer Agent — Screening Test Submission

**Track A — code-based agent.** Python + LangGraph.

```bash
python main.py --input test_data.xlsx
```

## What it does

Reads `test_data.xlsx` and runs four tasks automatically in one pipeline:

1. **Content generation** — for each of the 25 vendor SKUs, generates a marketplace-ready title, description, 5 key features, and structured attributes using Google Gemini (structured JSON output).
2. **Lifestyle images** — automated lifestyle image generation for product SKUs in modern living spaces. Runs 5 SKUs by default (cost control, as the brief allows); change `IMAGE_SUBSET` in `main.py` for more.
3. **Competitor analysis** — searches for comparable products via Tavily, then uses Gemini to pick the closest match and extract price/insight. Runs 8 SKUs by default, picking one per product type for variety.
4. **Sales analysis** — pure pandas, no API needed. Computes revenue, units, top/bottom SKUs, cancellation rate, carrier gaps, and business insights.

Pipeline flow (LangGraph StateGraph):
```
load_data → generate_content → generate_images → competitor_analysis → sales_analysis → write_output
```

Output goes to `output/` (JSON + CSV + markdown summary) and `images/` (one PNG per SKU).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in GEMINI_API_KEY (required) and TAVILY_API_KEY (for competitor research)
python main.py --input test_data.xlsx
```

## Data assumptions

**SKU mismatch is intentional.** The 25 vendor SKUs and the ~10,700 sales rows share zero common SKUs — confirmed programmatically at load time. Content/image/competitor work runs on the vendor sheet; sales analysis runs independently on the sales sheet.

**Product type extracted from Bullet 1.** The `Category` column is just "Furniture" for every row. The actual product type (power reclining chair, coffee table, TV stand, etc.) is parsed from the first bullet point with a simple regex.

**"Our price" is a modeled estimate.** Since there's no shared key to look up a real price, each vendor SKU is ranked by shipping weight (a standard furniture size/cost proxy) and mapped onto the real unit-price distribution from valid sales orders. Every competitor row's `our_price_method` field says this explicitly.

**Cancelled and Pending orders excluded** from revenue and units sold — they're not completed transactions. OJ Shipping is kept (order in fulfillment, not failed). Carrier Not Mapped rows (about 15%) are kept in revenue totals but flagged as a data gap.

## Files

```
main.py              Everything — data loading, content gen, images, competitors, sales, output
requirements.txt     Python dependencies
.env.example         API key template
test_data.xlsx       Input data (provided with the test)
output/              Generated output (content, competitor, sales — JSON + CSV + markdown)
images/              Generated lifestyle images, named by SKU
```
