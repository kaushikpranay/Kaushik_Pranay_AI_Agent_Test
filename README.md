AI Engineer Agent - Screening Test

Python + LangGraph implementation for the e-commerce screening task.

Run

python main.py --input test_data.xlsx

What it does

The pipeline takes the Excel file and performs four tasks:

1. Product content
   
   - Generates product title, description, 5 features, and attributes.
   - Uses Gemini for the content generation.

2. Lifestyle images
   
   - Uses the product image as a reference.
   - Generates lifestyle images for 5 products by default.
   - Change "IMAGE_SUBSET" in "main.py" to process more products.

3. Competitor analysis
   
   - Searches the web using Tavily.
   - Uses Gemini to find a comparable product and summarize the result.
   - Processes 8 products by default.

4. Sales analysis
   
   - Uses pandas only.
   - Calculates total sales, units sold, top SKUs, low-moving SKUs, and other sales information.

The pipeline is:

Load data
    ↓
Generate content
    ↓
Generate images
    ↓
Competitor analysis
    ↓
Sales analysis
    ↓
Write output

Setup

Install the dependencies:

pip install -r requirements.txt

Create the environment file:

cp .env.example .env

Add the required API keys:

GEMINI_API_KEY=your_key
TAVILY_API_KEY=your_key

Then run:

python main.py --input test_data.xlsx

Data handling

The vendor and sales sheets do not have matching SKUs, so they are processed separately.

For the vendor data, the product type is taken from the first bullet because the category column contains the same general category for all products.

There is no direct product price available for the vendor SKUs. For competitor analysis, the code therefore calculates an estimated price using product weight and the price distribution from the sales data. The output clearly marks this as an estimate.

For sales analysis:

- "Processed" and "OJ Shipping" orders are included.
- "Cancelled" and "Pending" orders are excluded.
- Orders with missing shipping carriers are still included in sales totals and reported separately.

Output

The program creates:

output/
├── content.json
├── content.csv
├── competitor_analysis.json
├── competitor_analysis.csv
├── sales_analysis.json
├── sales_summary.md
├── top_skus.csv
├── low_moving_skus.csv
├── images_manifest.json
└── run_log.json

images/
└── generated lifestyle images

Project structure

main.py
requirements.txt
.env.example
test_data.xlsx
output/
images/

The main logic is kept in "main.py" so the complete workflow can be run with a single command.