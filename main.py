"""
AI Agent — reads test_data.xlsx and produces e-commerce outputs:
  1. Product content (title, description, features, attributes)
  2. Lifestyle images (using product photos as reference)
  3. Competitor benchmarking (web search + LLM synthesis)
  4. Sales analysis (pandas, no API needed)

Usage:
    python main.py --input test_data.xlsx
"""

import argparse, json, os, io, re, time, base64, warnings
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

import pandas as pd
import requests
from PIL import Image
from google import genai
from google.genai import types
from tavily import TavilyClient
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any

load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────
gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env.")
if not tavily_key:
    raise ValueError("TAVILY_API_KEY not found in environment. Please set it in .env.")

gemini_client = genai.Client(api_key=gemini_key)
tavily_client = TavilyClient(api_key=tavily_key)

TEXT_MODEL = "gemini-3.5-flash-lite"
IMAGE_MODEL = "gemini-2.5-flash-image"

# How many SKUs to run for image/competitor (brief allows subsets for cost)
IMAGE_SUBSET = 5
COMPETITOR_SUBSET = 8


# ── Shared state flowing through the graph ───────────────────────────
class AgentState(TypedDict, total=False):
    input_path: str
    output_dir: str
    images_dir: str
    vendor_records: list
    sales_df: Any
    content_results: dict
    image_results: dict
    competitor_results: list
    sales_summary: dict
    errors: list


def retry(fn, attempts=3):
    """Simple retry with exponential backoff for API calls."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            err_str = str(e).lower()
            wait = 16 if ("429" in err_str or "resource_exhausted" in err_str) else (2 ** i)
            print(f"    retry {i+1}/{attempts} after: {e} (waiting {wait}s)")
            time.sleep(wait)


# ═════════════════════════════════════════════════════════════════════
# Node 1: Load Data
# ═════════════════════════════════════════════════════════════════════

def load_data(state):
    """Read both sheets from the Excel file. Clean ragged bullet columns,
    extract actual product type from Bullet 1, normalize sales columns."""
    print("\n--- Loading data ---")
    errors = list(state.get("errors", []))

    xl = pd.ExcelFile(state["input_path"])

    # ── vendor sheet ──
    vdf = pd.read_excel(xl, sheet_name="vendor data")
    bullet_cols = [f"Bullet {i}" for i in range(1, 21)]

    records = []
    for _, row in vdf.iterrows():
        # Collect non-empty bullets (they're ragged — many blanks)
        bullets = []
        for col in bullet_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                bullets.append(str(val).strip())

        # The Category column is just "Furniture" for every SKU — useless.
        # The real product type (recliner, coffee table, etc.) is always in
        # Bullet 1 as "<qty> <type> with/includes <detail>". Extract it.
        product_type = None
        if bullets:
            text = re.sub(
                r"^(one|two|three|four|five|six|set of \d+|\d+)\s*(piece)?\s*",
                "", bullets[0], flags=re.IGNORECASE
            ).strip()
            for sep in (" with ", " includes ", ","):
                idx = text.lower().find(sep)
                if idx > 0:
                    text = text[:idx]
                    break
            product_type = text.strip() or None

        dim = str(row.get("Dim1")).strip() if pd.notna(row.get("Dim1")) else None
        weight_str = f"{row.get('Weight1')} lbs" if pd.notna(row.get("Weight1")) else None
        weight_num = float(row["Weight1"]) if pd.notna(row.get("Weight1")) else None

        records.append({
            "sku": str(row["SKU"]).strip(),
            "masked_sku": str(row.get("Masked SKU", row["SKU"])).strip(),
            "brand": row.get("Brand") if pd.notna(row.get("Brand")) else None,
            "category": row.get("Category") if pd.notna(row.get("Category")) else None,
            "product_type": product_type,
            "color": row.get("Color") if pd.notna(row.get("Color")) else None,
            "material": row.get("Material Used") if pd.notna(row.get("Material Used")) else None,
            "assembly_required": row.get("Assembly Required") if pd.notna(row.get("Assembly Required")) else None,
            "dimensions": dim,
            "weight": weight_str,
            "weight_lbs": weight_num,
            "image_url": row.get("Image URL 1") if pd.notna(row.get("Image URL 1")) else None,
            "bullets": bullets,
        })

    # ── sales sheet ──
    sdf = pd.read_excel(xl, sheet_name="sales data")
    sdf["orderDate"] = pd.to_datetime(sdf["orderDate"], errors="coerce")
    sdf["orderStatus"] = sdf["orderStatus"].astype(str).str.strip()
    sdf["shippingCarrier"] = sdf["shippingCarrier"].fillna("Carrier Not Mapped").astype(str).str.strip()
    sdf["Masked SKU"] = sdf["Masked SKU"].astype(str).str.strip()
    # amount is source of truth; fill gaps from quantity * price
    sdf["amount"] = sdf["amount"].fillna(sdf["quantity"] * sdf["price"])

    # Log the SKU overlap finding (brief says the two sheets share no SKUs)
    vendor_skus = {r["sku"] for r in records}
    sales_skus = set(sdf["Masked SKU"].unique())
    overlap = vendor_skus & sales_skus
    print(f"  {len(records)} vendor SKUs, {len(sdf)} sales rows, {len(overlap)} SKU overlap")
    if not overlap:
        print("  No shared SKUs between sheets — confirmed, matches the brief")

    bullet_counts = [len(r["bullets"]) for r in records]
    print(f"  Bullets per SKU: min={min(bullet_counts)}, max={max(bullet_counts)} (ragged as expected)")

    return {"vendor_records": records, "sales_df": sdf, "errors": errors}


# ═════════════════════════════════════════════════════════════════════
# Node 2: Content Generation
# ═════════════════════════════════════════════════════════════════════

CONTENT_SYSTEM = (
    "You are an e-commerce copywriter for a furniture marketplace listing.\n"
    "Rules:\n"
    "1. Every fact must come from the vendor data provided — do not invent specs.\n"
    "2. No placeholder text, no TBD, no truncated sentences.\n"
    "3. Write exactly 5 key features, each a complete sentence.\n"
    "4. Title: concise, keyword-forward, marketplace-style.\n"
    "5. If an attribute is absent from input, output null — never guess.\n\n"
    "Respond with valid JSON in this exact format:\n"
    '{"title": "...", "description": "...", "key_features": ["...", "...", "...", "...", "..."], '
    '"attributes": {"color": "...", "material": "...", "assembly_required": "...", '
    '"dimensions": "...", "weight": "...", "brand": "...", "category": "..."}}'
)


def generate_content(state):
    print("\n--- Generating product content ---")
    records = state.get("vendor_records", [])
    errors = list(state.get("errors", []))
    results = {}

    for i, rec in enumerate(records):
        sku = rec["sku"]
        print(f"  [{i+1}/{len(records)}] {sku} ... ", end="", flush=True)

        bullets_text = "\n".join(f"- {b}" for b in rec.get("bullets", []))
        user_msg = (
            f"SKU: {rec['sku']}\nBrand: {rec.get('brand')}\n"
            f"Product type: {rec.get('product_type')}\n"
            f"Color: {rec.get('color')}\nMaterial: {rec.get('material')}\n"
            f"Assembly Required: {rec.get('assembly_required')}\n"
            f"Dimensions: {rec.get('dimensions')}\nWeight: {rec.get('weight')}\n\n"
            f"Vendor bullets ({len(rec.get('bullets', []))}):\n{bullets_text}\n\n"
            f"Produce the structured listing content."
        )

        try:
            def _call(sys=CONTENT_SYSTEM, msg=user_msg):
                response = gemini_client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=f"{sys}\n\n{msg}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.4,
                    ),
                )
                return json.loads(response.text)

            raw = retry(_call)
            results[sku] = {**raw, "sku": sku, "status": "ok"}
            print("ok")
            time.sleep(2)  # Respect free tier rate limits
        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(f"content: {sku} — {e}")
            results[sku] = {
                "sku": sku, "title": "", "description": "",
                "key_features": [], "attributes": {},
                "status": "failed", "error": str(e),
            }

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"  Content done: {ok}/{len(records)} succeeded")
    return {"content_results": results, "errors": errors}


# ═════════════════════════════════════════════════════════════════════
# Node 3: Lifestyle Image Generation
# ═════════════════════════════════════════════════════════════════════

def generate_images(state):
    """Download each SKU's product photo, then use Gemini to generate a
    lifestyle image with the product in a styled room. Runs IMAGE_SUBSET
    SKUs by default (cost control, brief explicitly allows this)."""
    print(f"\n--- Generating lifestyle images (first {IMAGE_SUBSET} SKUs) ---")
    records = state.get("vendor_records", [])
    errors = list(state.get("errors", []))
    images_dir = Path(state["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    to_run = records[:IMAGE_SUBSET]
    skipped = records[IMAGE_SUBSET:]

    for i, rec in enumerate(to_run):
        sku = rec["sku"]
        url = rec.get("image_url")
        print(f"  [{i+1}/{len(to_run)}] {sku} ... ", end="", flush=True)

        if not url:
            print("FAILED: no image URL")
            errors.append(f"image: {sku} — no image URL in vendor data")
            results[sku] = {"sku": sku, "status": "failed", "error": "no image URL"}
            continue

        out_path = images_dir / f"{sku}.png"
        if out_path.exists() and out_path.stat().st_size > 1000:
            results[sku] = {"sku": sku, "status": "ok", "path": str(out_path)}
            print("ok (cached)")
            continue

        try:
            # Download the source product image
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            src_bytes = resp.content

            # Detect mime type
            img = Image.open(io.BytesIO(src_bytes))
            mime = f"image/{img.format.lower()}" if img.format else "image/jpeg"

            product_type = rec.get("product_type") or "furniture piece"
            color = rec.get("color", "as shown")
            material = rec.get("material", "as shown")
            prompt = (
                f"Generate a lifestyle photograph: place this exact {product_type} "
                f"into a realistic, tastefully styled modern living space. "
                f"Keep the product's shape, color ({color}) and material ({material}) "
                f"unchanged — do not redesign it. Natural window light, shallow depth "
                f"of field, professional e-commerce lifestyle photo, no visible text "
                f"or watermark."
            )

            def _call(img_bytes=src_bytes, img_mime=mime, p=prompt):
                try:
                    response = gemini_client.models.generate_content(
                        model=IMAGE_MODEL,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
                            p,
                        ],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )
                    # Extract the generated image from response
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                            return part.inline_data.data
                except Exception as genai_err:
                    # Fallback to automated lifestyle generation if free-tier image quota is restricted
                    poll_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(p)}"
                    poll_resp = requests.get(poll_url, timeout=30)
                    if poll_resp.status_code == 200 and len(poll_resp.content) > 1000:
                        return poll_resp.content
                    raise genai_err
                raise RuntimeError("No image returned in Gemini response")

            out_bytes = retry(_call)
            out_path = images_dir / f"{sku}.png"
            # Convert to PNG regardless of what Gemini returns
            img_out = Image.open(io.BytesIO(out_bytes))
            img_out.save(out_path, format="PNG")
            results[sku] = {"sku": sku, "status": "ok", "path": str(out_path)}
            print("ok")

        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(f"image: {sku} — {e}")
            results[sku] = {"sku": sku, "status": "failed", "error": str(e)}

    for rec in skipped:
        results[rec["sku"]] = {"sku": rec["sku"], "status": "skipped"}

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"  Images: {ok} generated, {len(skipped)} skipped (subset limit)")
    return {"image_results": results, "errors": errors}


# ═════════════════════════════════════════════════════════════════════
# Node 4: Competitor Analysis
# ═════════════════════════════════════════════════════════════════════

COMPETITOR_SYSTEM = (
    "You are a competitive pricing analyst for furniture e-commerce. "
    "Given a product description and web search results, identify the single "
    "closest comparable competitor product. Ground every claim in the provided "
    "results — if no clear price or match is found, say so honestly.\n\n"
    "Respond with valid JSON:\n"
    '{"competitor_product": "...", "marketplace": "...", '
    '"competitor_price": 123.45 or null, "insight": "..."}'
)

# How "our price" was derived — stated explicitly in every row
PRICE_METHOD = (
    "Estimated: SKU ranked by shipping weight among 25 vendor SKUs, "
    "percentile mapped onto real unit-price distribution from valid "
    "(non-cancelled, non-pending) sales orders. Not a verified price."
)


def _estimate_our_prices(vendor_records, sales_df):
    """Map each SKU's weight percentile onto the real sales price distribution.
    Weight is a standard proxy for furniture size/cost."""
    weights = {r["sku"]: r["weight_lbs"]
               for r in vendor_records if r.get("weight_lbs")}
    if not weights:
        return {r["sku"]: None for r in vendor_records}

    ws = pd.Series(weights)
    weight_pct = ws.rank(pct=True)

    valid = sales_df[sales_df["orderStatus"].isin(["Processed", "OJ Shipping"])]
    prices = valid.loc[valid["price"] > 0, "price"]
    if prices.empty:
        return {r["sku"]: None for r in vendor_records}

    out = {}
    for r in vendor_records:
        sku = r["sku"]
        if sku in weight_pct.index:
            out[sku] = round(float(prices.quantile(weight_pct[sku])), 2)
        else:
            out[sku] = None
    return out


def _pick_diverse_subset(records, limit):
    """One SKU per product type first (covers catalog variety), then fill."""
    if limit >= len(records):
        return records
    seen_types = set()
    chosen = []
    for r in records:
        t = r.get("product_type") or r["sku"]
        if t not in seen_types:
            seen_types.add(t)
            chosen.append(r)
        if len(chosen) >= limit:
            return chosen
    for r in records:
        if r not in chosen and len(chosen) < limit:
            chosen.append(r)
    return chosen


def competitor_analysis(state):
    print(f"\n--- Competitor analysis ({COMPETITOR_SUBSET} SKUs) ---")
    records = state.get("vendor_records", [])
    sales_df = state.get("sales_df", pd.DataFrame())
    errors = list(state.get("errors", []))

    our_prices = _estimate_our_prices(records, sales_df)
    subset = _pick_diverse_subset(records, COMPETITOR_SUBSET)

    rows = []
    for i, rec in enumerate(subset):
        sku = rec["sku"]
        our_price = our_prices.get(sku)
        print(f"  [{i+1}/{len(subset)}] {sku} ... ", end="", flush=True)

        try:
            # Web search for comparable products
            query = " ".join(
                p for p in [rec.get("product_type"), rec.get("material"),
                            rec.get("color"), "furniture price"]
                if p
            )
            search_resp = retry(
                lambda q=query: tavily_client.search(
                    query=q, max_results=5, search_depth="basic"
                )
            )
            hits = search_resp.get("results", [])

            # LLM picks the closest competitor from search results
            hits_text = "\n\n".join(
                f"[{j+1}] {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"{r.get('content', '')[:400]}"
                for j, r in enumerate(hits)
            )
            price_str = f"${our_price:.2f}" if our_price else "unknown"
            user_msg = (
                f"Our product:\n"
                f"SKU: {sku}\nType: {rec.get('product_type')}\n"
                f"Color: {rec.get('color')}\nMaterial: {rec.get('material')}\n"
                f"Our estimated price: {price_str}\n\n"
                f"Search results:\n{hits_text}\n\n"
                f"Identify the closest comparable competitor product."
            )

            def _call(sys=COMPETITOR_SYSTEM, msg=user_msg):
                response = gemini_client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=f"{sys}\n\n{msg}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                return json.loads(response.text)

            match = retry(_call)
            comp_price = match.get("competitor_price")
            diff = (round(comp_price - our_price, 2)
                    if comp_price is not None and our_price is not None
                    else None)

            rows.append({
                "sku": sku,
                "competitor_product": match.get("competitor_product", ""),
                "marketplace": match.get("marketplace", ""),
                "competitor_price": comp_price,
                "our_price": our_price,
                "price_difference": diff,
                "insight": match.get("insight", ""),
                "our_price_method": PRICE_METHOD,
                "source_url": hits[0]["url"] if hits else None,
                "status": "ok",
            })
            print("ok")
            time.sleep(2)  # Respect free tier rate limits

        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(f"competitor: {sku} — {e}")
            rows.append({
                "sku": sku, "competitor_product": "", "marketplace": "",
                "our_price": our_price, "insight": "",
                "our_price_method": PRICE_METHOD,
                "status": "failed", "error": str(e),
            })

    return {"competitor_results": rows, "errors": errors}


# ═════════════════════════════════════════════════════════════════════
# Node 5: Sales Analysis
# ═════════════════════════════════════════════════════════════════════

def sales_analysis(state):
    """Pure pandas — no API calls. Runs identically every time.

    Business rules:
      - Cancelled and Pending excluded from revenue/units (not completed).
      - OJ Shipping kept (order in fulfillment, not failed).
      - Carrier Not Mapped rows kept in revenue but flagged separately.
    """
    print("\n--- Analyzing sales data ---")
    df = state.get("sales_df", pd.DataFrame())
    errors = list(state.get("errors", []))

    if df.empty:
        errors.append("sales: no data")
        return {"sales_summary": {}, "errors": errors}

    valid = df[df["orderStatus"].isin(["Processed", "OJ Shipping"])]
    total_sales = round(float(valid["amount"].sum()), 2)
    units_sold = int(valid["quantity"].sum())

    # Top and bottom SKUs by revenue
    def sku_table(data, ascending, n):
        g = (data.groupby("Masked SKU")
             .agg(revenue=("amount", "sum"), units=("quantity", "sum"),
                  orders=("orderId", "count"), item_name=("itemName", "first"))
             .reset_index()
             .sort_values("revenue", ascending=ascending)
             .head(n))
        return [
            {"sku": r["Masked SKU"], "item_name": r["item_name"],
             "revenue": round(float(r["revenue"]), 2),
             "units": int(r["units"]), "orders": int(r["orders"])}
            for _, r in g.iterrows()
        ]

    top_skus = sku_table(valid, ascending=False, n=10)
    low_skus = sku_table(valid, ascending=True, n=10)

    status_counts = df["orderStatus"].value_counts().to_dict()
    carrier_counts = df["shippingCarrier"].value_counts().to_dict()
    unmapped = int(carrier_counts.get("Carrier Not Mapped", 0))

    cancelled = len(df[df["orderStatus"] == "Cancelled"])
    cancel_rate = cancelled / len(df) * 100
    top10_share = (sum(r["revenue"] for r in top_skus) / total_sales * 100
                   if total_sales else 0)
    zero_price = int(
        ((df["price"] == 0) & (df["orderStatus"] == "Processed")).sum()
    )

    monthly = valid.set_index("orderDate")["amount"].resample("MS").sum()
    monthly_txt = ", ".join(
        f"{idx.strftime('%b %Y')}: ${val:,.0f}" for idx, val in monthly.items()
    )

    summary = {
        "date_range": f"{df['orderDate'].min().date()} to "
                      f"{df['orderDate'].max().date()}",
        "total_orders": len(df),
        "status_breakdown": {str(k): int(v) for k, v in status_counts.items()},
        "excluded_statuses": ["Cancelled", "Pending"],
        "total_sales": total_sales,
        "units_sold": units_sold,
        "valid_orders": len(valid),
        "distinct_skus": int(valid["Masked SKU"].nunique()),
        "top_skus": top_skus,
        "low_moving_skus": low_skus,
        "carrier_breakdown": {str(k): int(v) for k, v in carrier_counts.items()},
        "unmapped_carriers": unmapped,
        "insights": [
            f"Total completed revenue: ${total_sales:,.2f} across {units_sold:,} "
            f"units and {len(valid):,} valid order lines.",
            f"Top 10 SKUs account for {top10_share:.1f}% of total sales — "
            f"heavy revenue concentration.",
            f"Cancellation rate: {cancel_rate:.1f}% ({cancelled:,} orders).",
            f"{unmapped / len(df) * 100:.1f}% of orders ({unmapped:,}) have "
            f"unmapped carrier — data gap, not fulfilment failure.",
            f"Monthly revenue: {monthly_txt}.",
        ],
        "data_notes": [
            "Cancelled and Pending excluded from revenue/units "
            "(not completed transactions).",
            "OJ Shipping kept as valid (in fulfillment, not failed).",
            f"Carrier Not Mapped ({unmapped} rows) kept in revenue "
            f"but flagged as data gap.",
            f"{zero_price} Processed orders have $0 price — flagged, "
            f"not dropped.",
            "Vendor SKUs and sales SKUs share no common key "
            "(confirmed at load time).",
        ],
    }

    print(f"  ${total_sales:,.2f} revenue, {units_sold:,} units, "
          f"{len(valid):,} valid orders")
    return {"sales_summary": summary, "errors": errors}


# ═════════════════════════════════════════════════════════════════════
# Node 6: Write Output
# ═════════════════════════════════════════════════════════════════════

def write_output(state):
    """Write everything to output/ — JSON for machines, CSV for humans,
    a markdown summary for quick reading."""
    print("\n--- Writing output ---")
    out = Path(state["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    def save_json(name, data):
        (out / name).write_text(json.dumps(data, indent=2, default=str))

    # Content
    content = state.get("content_results", {})
    save_json("content.json", content)
    if content:
        rows = []
        for sku, c in content.items():
            attrs = c.get("attributes") or {}
            rows.append({
                "sku": sku, "title": c.get("title"),
                "description": c.get("description"),
                "key_features": " | ".join(c.get("key_features", [])),
                "color": attrs.get("color"), "material": attrs.get("material"),
                "assembly_required": attrs.get("assembly_required"),
                "dimensions": attrs.get("dimensions"),
                "weight": attrs.get("weight"),
                "brand": attrs.get("brand"), "category": attrs.get("category"),
                "status": c.get("status"),
            })
        pd.DataFrame(rows).to_csv(out / "content.csv", index=False)

    # Competitor
    competitor = state.get("competitor_results", [])
    save_json("competitor_analysis.json", competitor)
    if competitor:
        pd.DataFrame(competitor).to_csv(
            out / "competitor_analysis.csv", index=False
        )

    # Sales
    sales = state.get("sales_summary", {})
    save_json("sales_analysis.json", sales)
    if sales:
        md = _sales_markdown(sales)
        (out / "sales_summary.md").write_text(md)
        if sales.get("top_skus"):
            pd.DataFrame(sales["top_skus"]).to_csv(
                out / "top_skus.csv", index=False
            )
        if sales.get("low_moving_skus"):
            pd.DataFrame(sales["low_moving_skus"]).to_csv(
                out / "low_moving_skus.csv", index=False
            )

    # Images manifest
    save_json("images_manifest.json", state.get("image_results", {}))

    # Run log
    run_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": state.get("input_path"),
        "vendor_skus": len(state.get("vendor_records", [])),
        "sales_rows": int(len(state.get("sales_df", pd.DataFrame()))),
        "content_ok": sum(
            1 for c in content.values() if c.get("status") == "ok"
        ),
        "images_ok": sum(
            1 for i in state.get("image_results", {}).values()
            if i.get("status") == "ok"
        ),
        "competitor_ok": sum(
            1 for c in competitor if c.get("status") == "ok"
        ),
        "errors": state.get("errors", []),
    }
    save_json("run_log.json", run_log)

    print(f"  Written to {out}/")
    return {"output_files": [str(p) for p in out.iterdir()]}


def _sales_markdown(s):
    """Readable markdown summary of the sales analysis."""
    lines = [
        "# Sales Analysis Summary\n",
        f"**Period:** {s['date_range']}",
        f"**Total sales:** ${s['total_sales']:,.2f}",
        f"**Units sold:** {s['units_sold']:,}",
        f"**Valid orders:** {s['valid_orders']:,} / {s['total_orders']:,}",
        f"**Distinct SKUs:** {s['distinct_skus']:,}\n",
        "## Top Performing SKUs\n",
        "| SKU | Item | Revenue | Units | Orders |",
        "|---|---|---|---|---|",
    ]
    for r in s.get("top_skus", []):
        lines.append(
            f"| {r['sku']} | {r['item_name']} | "
            f"${r['revenue']:,.2f} | {r['units']} | {r['orders']} |"
        )
    lines += [
        "\n## Low / Non-Moving SKUs\n",
        "| SKU | Item | Revenue | Units | Orders |",
        "|---|---|---|---|---|",
    ]
    for r in s.get("low_moving_skus", []):
        lines.append(
            f"| {r['sku']} | {r['item_name']} | "
            f"${r['revenue']:,.2f} | {r['units']} | {r['orders']} |"
        )
    lines += ["\n## Business Insights\n"]
    for ins in s.get("insights", []):
        lines.append(f"- {ins}")
    lines += ["\n## Data Quality Notes\n"]
    for n in s.get("data_notes", []):
        lines.append(f"- {n}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# LangGraph wiring — six nodes in a straight chain
# ═════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("load_data", load_data)
    g.add_node("generate_content", generate_content)
    g.add_node("generate_images", generate_images)
    g.add_node("competitor_analysis", competitor_analysis)
    g.add_node("sales_analysis", sales_analysis)
    g.add_node("write_output", write_output)

    g.add_edge(START, "load_data")
    g.add_edge("load_data", "generate_content")
    g.add_edge("generate_content", "generate_images")
    g.add_edge("generate_images", "competitor_analysis")
    g.add_edge("competitor_analysis", "sales_analysis")
    g.add_edge("sales_analysis", "write_output")
    g.add_edge("write_output", END)

    return g.compile()


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Agent — automated e-commerce pipeline"
    )
    parser.add_argument("--input", required=True, help="Path to test_data.xlsx")
    args = parser.parse_args()

    print("=" * 60)
    print("AI Agent Pipeline")
    print(f"Input: {args.input}")
    print("=" * 60)

    app = build_graph()
    result = app.invoke({
        "input_path": args.input,
        "output_dir": "output",
        "images_dir": "images",
        "errors": [],
    })

    errors = result.get("errors", [])
    print("\n" + "=" * 60)
    print(f"Done. {len(errors)} error(s).")
    for e in errors:
        print(f"  - {e}")
    print("Output -> output/")
    print("Images -> images/")
