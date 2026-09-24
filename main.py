import argparse
import io
import json
import os
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient
from typing import Any, TypedDict

warnings.filterwarnings("ignore")
load_dotenv()

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

IMAGE_SUBSET = 5
COMPETITOR_SUBSET = 8


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
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise

            error = str(e).lower()
            wait = 16 if "429" in error or "resource_exhausted" in error else 2 ** i
            print(f"    retry {i + 1}/{attempts} after: {e} (waiting {wait}s)")
            time.sleep(wait)


def load_data(state):
    print("\n--- Loading data ---")

    errors = list(state.get("errors", []))
    xl = pd.ExcelFile(state["input_path"])

    vendor_df = pd.read_excel(xl, sheet_name="vendor data")
    bullet_cols = [f"Bullet {i}" for i in range(1, 21)]

    records = []

    for _, row in vendor_df.iterrows():
        bullets = []

        for col in bullet_cols:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                bullets.append(str(value).strip())

        product_type = None

        if bullets:
            text = re.sub(
                r"^(one|two|three|four|five|six|set of \d+|\d+)\s*(piece)?\s*",
                "",
                bullets[0],
                flags=re.IGNORECASE,
            ).strip()

            for separator in (" with ", " includes ", ","):
                index = text.lower().find(separator)
                if index > 0:
                    text = text[:index]
                    break

            product_type = text.strip() or None

        dimensions = (
            str(row.get("Dim1")).strip()
            if pd.notna(row.get("Dim1"))
            else None
        )

        weight = row.get("Weight1")
        weight_str = f"{weight} lbs" if pd.notna(weight) else None
        weight_num = float(weight) if pd.notna(weight) else None

        records.append({
            "sku": str(row["SKU"]).strip(),
            "masked_sku": str(row.get("Masked SKU", row["SKU"])).strip(),
            "brand": row.get("Brand") if pd.notna(row.get("Brand")) else None,
            "category": row.get("Category") if pd.notna(row.get("Category")) else None,
            "product_type": product_type,
            "color": row.get("Color") if pd.notna(row.get("Color")) else None,
            "material": (
                row.get("Material Used")
                if pd.notna(row.get("Material Used"))
                else None
            ),
            "assembly_required": (
                row.get("Assembly Required")
                if pd.notna(row.get("Assembly Required"))
                else None
            ),
            "dimensions": dimensions,
            "weight": weight_str,
            "weight_lbs": weight_num,
            "image_url": (
                row.get("Image URL 1")
                if pd.notna(row.get("Image URL 1"))
                else None
            ),
            "bullets": bullets,
        })

    sales_df = pd.read_excel(xl, sheet_name="sales data")
    sales_df["orderDate"] = pd.to_datetime(
        sales_df["orderDate"], errors="coerce"
    )
    sales_df["orderStatus"] = sales_df["orderStatus"].astype(str).str.strip()
    sales_df["shippingCarrier"] = (
        sales_df["shippingCarrier"]
        .fillna("Carrier Not Mapped")
        .astype(str)
        .str.strip()
    )
    sales_df["Masked SKU"] = sales_df["Masked SKU"].astype(str).str.strip()
    sales_df["amount"] = sales_df["amount"].fillna(
        sales_df["quantity"] * sales_df["price"]
    )

    vendor_skus = {r["sku"] for r in records}
    sales_skus = set(sales_df["Masked SKU"].unique())
    overlap = vendor_skus & sales_skus

    print(
        f"  {len(records)} vendor SKUs, "
        f"{len(sales_df)} sales rows, {len(overlap)} SKU overlap"
    )

    if not overlap:
        print("  No shared SKUs between sheets")

    bullet_counts = [len(r["bullets"]) for r in records]
    print(
        f"  Bullets per SKU: min={min(bullet_counts)}, "
        f"max={max(bullet_counts)}"
    )

    return {
        "vendor_records": records,
        "sales_df": sales_df,
        "errors": errors,
    }


CONTENT_SYSTEM = (
    "You are an e-commerce copywriter for a furniture marketplace listing.\n"
    "Rules:\n"
    "1. Every fact must come from the vendor data provided — do not invent specs.\n"
    "2. No placeholder text, no TBD, no truncated sentences.\n"
    "3. Write exactly 5 key features, each a complete sentence.\n"
    "4. Title: concise, keyword-forward, marketplace-style.\n"
    "5. If an attribute is absent from input, output null — never guess.\n\n"
    "Respond with valid JSON in this exact format:\n"
    '{"title": "...", "description": "...", '
    '"key_features": ["...", "...", "...", "...", "..."], '
    '"attributes": {"color": "...", "material": "...", '
    '"assembly_required": "...", "dimensions": "...", '
    '"weight": "...", "brand": "...", "category": "..."}}'
)


def generate_content(state):
    print("\n--- Generating product content ---")

    records = state.get("vendor_records", [])
    errors = list(state.get("errors", []))
    results = {}

    for i, rec in enumerate(records):
        sku = rec["sku"]
        print(f"  [{i + 1}/{len(records)}] {sku} ... ", end="", flush=True)

        bullets_text = "\n".join(
            f"- {bullet}" for bullet in rec.get("bullets", [])
        )

        user_msg = (
            f"SKU: {rec['sku']}\n"
            f"Brand: {rec.get('brand')}\n"
            f"Product type: {rec.get('product_type')}\n"
            f"Color: {rec.get('color')}\n"
            f"Material: {rec.get('material')}\n"
            f"Assembly Required: {rec.get('assembly_required')}\n"
            f"Dimensions: {rec.get('dimensions')}\n"
            f"Weight: {rec.get('weight')}\n\n"
            f"Vendor bullets ({len(rec.get('bullets', []))}):\n"
            f"{bullets_text}\n\n"
            "Produce the structured listing content."
        )

        try:
            def call_model():
                response = gemini_client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=f"{CONTENT_SYSTEM}\n\n{user_msg}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.4,
                    ),
                )
                return json.loads(response.text)

            result = retry(call_model)

            results[sku] = {
                **result,
                "sku": sku,
                "status": "ok",
            }

            print("ok")
            time.sleep(2)

        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(f"content: {sku} — {e}")

            results[sku] = {
                "sku": sku,
                "title": "",
                "description": "",
                "key_features": [],
                "attributes": {},
                "status": "failed",
                "error": str(e),
            }

    success = sum(
        1 for result in results.values()
        if result["status"] == "ok"
    )

    print(f"  Content done: {success}/{len(records)} succeeded")

    return {
        "content_results": results,
        "errors": errors,
    }


def generate_images(state):
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

        print(f"  [{i + 1}/{len(to_run)}] {sku} ... ", end="", flush=True)

        if not url:
            print("FAILED: no image URL")
            errors.append(f"image: {sku} — no image URL in vendor data")
            results[sku] = {
                "sku": sku,
                "status": "failed",
                "error": "no image URL",
            }
            continue

        out_path = images_dir / f"{sku}.png"

        if out_path.exists() and out_path.stat().st_size > 1000:
            results[sku] = {
                "sku": sku,
                "status": "ok",
                "path": str(out_path),
            }
            print("ok (cached)")
            continue

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            source_bytes = response.content

            image = Image.open(io.BytesIO(source_bytes))
            mime = (
                f"image/{image.format.lower()}"
                if image.format
                else "image/jpeg"
            )

            product_type = rec.get("product_type") or "furniture piece"
            color = rec.get("color", "as shown")
            material = rec.get("material", "as shown")

            prompt = (
                f"Generate a lifestyle photograph: place this exact "
                f"{product_type} into a realistic, tastefully styled "
                f"modern living space. Keep the product's shape, color "
                f"({color}) and material ({material}) unchanged — do not "
                "redesign it. Natural window light, shallow depth of field, "
                "professional e-commerce lifestyle photo, no visible text "
                "or watermark."
            )

            def generate_image():
                try:
                    response = gemini_client.models.generate_content(
                        model=IMAGE_MODEL,
                        contents=[
                            types.Part.from_bytes(
                                data=source_bytes,
                                mime_type=mime,
                            ),
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )

                    for part in response.candidates[0].content.parts:
                        if (
                            part.inline_data
                            and part.inline_data.mime_type.startswith("image/")
                        ):
                            return part.inline_data.data

                except Exception as genai_error:
                    fallback_url = (
                        "https://image.pollinations.ai/prompt/"
                        f"{requests.utils.quote(prompt)}"
                    )

                    fallback_response = requests.get(
                        fallback_url,
                        timeout=30,
                    )

                    if (
                        fallback_response.status_code == 200
                        and len(fallback_response.content) > 1000
                    ):
                        return fallback_response.content

                    raise genai_error

                raise RuntimeError("No image returned in Gemini response")

            output_bytes = retry(generate_image)

            output_image = Image.open(io.BytesIO(output_bytes))
            output_image.save(out_path, format="PNG")

            results[sku] = {
                "sku": sku,
                "status": "ok",
                "path": str(out_path),
            }

            print("ok")

        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(f"image: {sku} — {e}")

            results[sku] = {
                "sku": sku,
                "status": "failed",
                "error": str(e),
            }

    for rec in skipped:
        results[rec["sku"]] = {
            "sku": rec["sku"],
            "status": "skipped",
        }

    success = sum(
        1 for result in results.values()
        if result["status"] == "ok"
    )

    print(
        f"  Images: {success} generated, "
        f"{len(skipped)} skipped (subset limit)"
    )

    return {
        "image_results": results,
        "errors": errors,
    }


COMPETITOR_SYSTEM = (
    "You are a competitive pricing analyst for furniture e-commerce. "
    "Given a product description and web search results, identify the "
    "single closest comparable competitor product. Ground every claim in "
    "the provided results — if no clear price or match is found, say so "
    "honestly.\n\n"
    "Respond with valid JSON:\n"
    '{"competitor_product": "...", "marketplace": "...", '
    '"competitor_price": 123.45 or null, "insight": "..."}'
)

PRICE_METHOD = (
    "Estimated: SKU ranked by shipping weight among 25 vendor SKUs, "
    "percentile mapped onto real unit-price distribution from valid "
    "(non-cancelled, non-pending) sales orders. Not a verified price."
)


def estimate_our_prices(vendor_records, sales_df):
    weights = {
        record["sku"]: record["weight_lbs"]
        for record in vendor_records
        if record.get("weight_lbs")
    }

    if not weights:
        return {record["sku"]: None for record in vendor_records}

    weight_series = pd.Series(weights)
    weight_percentile = weight_series.rank(pct=True)

    valid = sales_df[
        sales_df["orderStatus"].isin(["Processed", "OJ Shipping"])
    ]

    prices = valid.loc[valid["price"] > 0, "price"]

    if prices.empty:
        return {record["sku"]: None for record in vendor_records}

    result = {}

    for record in vendor_records:
        sku = record["sku"]

        if sku in weight_percentile.index:
            result[sku] = round(
                float(prices.quantile(weight_percentile[sku])),
                2,
            )
        else:
            result[sku] = None

    return result


def pick_diverse_subset(records, limit):
    if limit >= len(records):
        return records

    seen_types = set()
    selected = []

    for record in records:
        product_type = record.get("product_type") or record["sku"]

        if product_type not in seen_types:
            seen_types.add(product_type)
            selected.append(record)

        if len(selected) >= limit:
            return selected

    for record in records:
        if record not in selected and len(selected) < limit:
            selected.append(record)

    return selected


def competitor_analysis(state):
    print(f"\n--- Competitor analysis ({COMPETITOR_SUBSET} SKUs) ---")

    records = state.get("vendor_records", [])
    sales_df = state.get("sales_df", pd.DataFrame())
    errors = list(state.get("errors", []))

    our_prices = estimate_our_prices(records, sales_df)
    subset = pick_diverse_subset(records, COMPETITOR_SUBSET)

    rows = []

    for i, rec in enumerate(subset):
        sku = rec["sku"]
        our_price = our_prices.get(sku)

        print(
            f"  [{i + 1}/{len(subset)}] {sku} ... ",
            end="",
            flush=True,
        )

        try:
            query = " ".join(
                value
                for value in [
                    rec.get("product_type"),
                    rec.get("material"),
                    rec.get("color"),
                    "furniture price",
                ]
                if value
            )

            search_response = retry(
                lambda q=query: tavily_client.search(
                    query=q,
                    max_results=5,
                    search_depth="basic",
                )
            )

            hits = search_response.get("results", [])

            hits_text = "\n\n".join(
                f"[{j + 1}] {result.get('title', '')}\n"
                f"URL: {result.get('url', '')}\n"
                f"{result.get('content', '')[:400]}"
                for j, result in enumerate(hits)
            )

            price_text = (
                f"${our_price:.2f}"
                if our_price
                else "unknown"
            )

            user_msg = (
                "Our product:\n"
                f"SKU: {sku}\n"
                f"Type: {rec.get('product_type')}\n"
                f"Color: {rec.get('color')}\n"
                f"Material: {rec.get('material')}\n"
                f"Our estimated price: {price_text}\n\n"
                f"Search results:\n{hits_text}\n\n"
                "Identify the closest comparable competitor product."
            )

            def call_model():
                response = gemini_client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=f"{COMPETITOR_SYSTEM}\n\n{user_msg}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                return json.loads(response.text)

            match = retry(call_model)

            competitor_price = match.get("competitor_price")

            price_difference = (
                round(competitor_price - our_price, 2)
                if competitor_price is not None and our_price is not None
                else None
            )

            rows.append({
                "sku": sku,
                "competitor_product": match.get("competitor_product", ""),
                "marketplace": match.get("marketplace", ""),
                "competitor_price": competitor_price,
                "our_price": our_price,
                "price_difference": price_difference,
                "insight": match.get("insight", ""),
                "our_price_method": PRICE_METHOD,
                "source_url": hits[0]["url"] if hits else None,
                "status": "ok",
            })

            print("ok")
            time.sleep(2)

        except Exception as e:
            print(f"FAILED: {e}")
            errors.append(f"competitor: {sku} — {e}")

            rows.append({
                "sku": sku,
                "competitor_product": "",
                "marketplace": "",
                "our_price": our_price,
                "insight": "",
                "our_price_method": PRICE_METHOD,
                "status": "failed",
                "error": str(e),
            })

    return {
        "competitor_results": rows,
        "errors": errors,
    }


def sales_analysis(state):
    print("\n--- Analyzing sales data ---")

    df = state.get("sales_df", pd.DataFrame())
    errors = list(state.get("errors", []))

    if df.empty:
        errors.append("sales: no data")
        return {
            "sales_summary": {},
            "errors": errors,
        }

    valid = df[
        df["orderStatus"].isin(["Processed", "OJ Shipping"])
    ]

    total_sales = round(float(valid["amount"].sum()), 2)
    units_sold = int(valid["quantity"].sum())

    def sku_table(data, ascending, n):
        grouped = (
            data.groupby("Masked SKU")
            .agg(
                revenue=("amount", "sum"),
                units=("quantity", "sum"),
                orders=("orderId", "count"),
                item_name=("itemName", "first"),
            )
            .reset_index()
            .sort_values("revenue", ascending=ascending)
            .head(n)
        )

        return [
            {
                "sku": row["Masked SKU"],
                "item_name": row["item_name"],
                "revenue": round(float(row["revenue"]), 2),
                "units": int(row["units"]),
                "orders": int(row["orders"]),
            }
            for _, row in grouped.iterrows()
        ]

    top_skus = sku_table(valid, ascending=False, n=10)
    low_skus = sku_table(valid, ascending=True, n=10)

    status_counts = df["orderStatus"].value_counts().to_dict()
    carrier_counts = df["shippingCarrier"].value_counts().to_dict()

    unmapped = int(
        carrier_counts.get("Carrier Not Mapped", 0)
    )

    cancelled = len(df[df["orderStatus"] == "Cancelled"])
    cancel_rate = cancelled / len(df) * 100

    top10_share = (
        sum(row["revenue"] for row in top_skus) / total_sales * 100
        if total_sales
        else 0
    )

    zero_price = int(
        (
            (df["price"] == 0)
            & (df["orderStatus"] == "Processed")
        ).sum()
    )

    monthly = (
        valid.set_index("orderDate")["amount"]
        .resample("MS")
        .sum()
    )

    monthly_text = ", ".join(
        f"{index.strftime('%b %Y')}: ${value:,.0f}"
        for index, value in monthly.items()
    )

    summary = {
        "date_range": (
            f"{df['orderDate'].min().date()} to "
            f"{df['orderDate'].max().date()}"
        ),
        "total_orders": len(df),
        "status_breakdown": {
            str(key): int(value)
            for key, value in status_counts.items()
        },
        "excluded_statuses": ["Cancelled", "Pending"],
        "total_sales": total_sales,
        "units_sold": units_sold,
        "valid_orders": len(valid),
        "distinct_skus": int(valid["Masked SKU"].nunique()),
        "top_skus": top_skus,
        "low_moving_skus": low_skus,
        "carrier_breakdown": {
            str(key): int(value)
            for key, value in carrier_counts.items()
        },
        "unmapped_carriers": unmapped,
        "insights": [
            f"Total completed revenue: ${total_sales:,.2f} "
            f"across {units_sold:,} units and {len(valid):,} valid "
            "order lines.",
            f"Top 10 SKUs account for {top10_share:.1f}% of total sales "
            "— heavy revenue concentration.",
            f"Cancellation rate: {cancel_rate:.1f}% "
            f"({cancelled:,} orders).",
            f"{unmapped / len(df) * 100:.1f}% of orders ({unmapped:,}) "
            "have unmapped carrier — data gap, not fulfilment failure.",
            f"Monthly revenue: {monthly_text}.",
        ],
        "data_notes": [
            "Cancelled and Pending excluded from revenue/units "
            "(not completed transactions).",
            "OJ Shipping kept as valid (in fulfillment, not failed).",
            f"Carrier Not Mapped ({unmapped} rows) kept in revenue "
            "but flagged as data gap.",
            f"{zero_price} Processed orders have $0 price — flagged, "
            "not dropped.",
            "Vendor SKUs and sales SKUs share no common key "
            "(confirmed at load time).",
        ],
    }

    print(
        f"  ${total_sales:,.2f} revenue, "
        f"{units_sold:,} units, {len(valid):,} valid orders"
    )

    return {
        "sales_summary": summary,
        "errors": errors,
    }


def sales_markdown(summary):
    lines = [
        "# Sales Analysis Summary\n",
        f"**Period:** {summary['date_range']}",
        f"**Total sales:** ${summary['total_sales']:,.2f}",
        f"**Units sold:** {summary['units_sold']:,}",
        f"**Valid orders:** "
        f"{summary['valid_orders']:,} / {summary['total_orders']:,}",
        f"**Distinct SKUs:** {summary['distinct_skus']:,}\n",
        "## Top Performing SKUs\n",
        "| SKU | Item | Revenue | Units | Orders |",
        "|---|---|---|---|---|",
    ]

    for row in summary.get("top_skus", []):
        lines.append(
            f"| {row['sku']} | {row['item_name']} | "
            f"${row['revenue']:,.2f} | {row['units']} | "
            f"{row['orders']} |"
        )

    lines += [
        "\n## Low / Non-Moving SKUs\n",
        "| SKU | Item | Revenue | Units | Orders |",
        "|---|---|---|---|---|",
    ]

    for row in summary.get("low_moving_skus", []):
        lines.append(
            f"| {row['sku']} | {row['item_name']} | "
            f"${row['revenue']:,.2f} | {row['units']} | "
            f"{row['orders']} |"
        )

    lines.append("\n## Business Insights\n")

    for insight in summary.get("insights", []):
        lines.append(f"- {insight}")

    lines.append("\n## Data Quality Notes\n")

    for note in summary.get("data_notes", []):
        lines.append(f"- {note}")

    return "\n".join(lines)


def write_output(state):
    print("\n--- Writing output ---")

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(name, data):
        (output_dir / name).write_text(
            json.dumps(data, indent=2, default=str)
        )

    content = state.get("content_results", {})
    save_json("content.json", content)

    if content:
        rows = []

        for sku, result in content.items():
            attributes = result.get("attributes") or {}

            rows.append({
                "sku": sku,
                "title": result.get("title"),
                "description": result.get("description"),
                "key_features": " | ".join(
                    result.get("key_features", [])
                ),
                "color": attributes.get("color"),
                "material": attributes.get("material"),
                "assembly_required": attributes.get("assembly_required"),
                "dimensions": attributes.get("dimensions"),
                "weight": attributes.get("weight"),
                "brand": attributes.get("brand"),
                "category": attributes.get("category"),
                "status": result.get("status"),
            })

        pd.DataFrame(rows).to_csv(
            output_dir / "content.csv",
            index=False,
        )

    competitor = state.get("competitor_results", [])
    save_json("competitor_analysis.json", competitor)

    if competitor:
        pd.DataFrame(competitor).to_csv(
            output_dir / "competitor_analysis.csv",
            index=False,
        )

    sales = state.get("sales_summary", {})
    save_json("sales_analysis.json", sales)

    if sales:
        (output_dir / "sales_summary.md").write_text(
            sales_markdown(sales)
        )

        if sales.get("top_skus"):
            pd.DataFrame(sales["top_skus"]).to_csv(
                output_dir / "top_skus.csv",
                index=False,
            )

        if sales.get("low_moving_skus"):
            pd.DataFrame(sales["low_moving_skus"]).to_csv(
                output_dir / "low_moving_skus.csv",
                index=False,
            )

    save_json(
        "images_manifest.json",
        state.get("image_results", {}),
    )

    run_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": state.get("input_path"),
        "vendor_skus": len(state.get("vendor_records", [])),
        "sales_rows": int(
            len(state.get("sales_df", pd.DataFrame()))
        ),
        "content_ok": sum(
            1
            for result in content.values()
            if result.get("status") == "ok"
        ),
        "images_ok": sum(
            1
            for result in state.get("image_results", {}).values()
            if result.get("status") == "ok"
        ),
        "competitor_ok": sum(
            1
            for result in competitor
            if result.get("status") == "ok"
        ),
        "errors": state.get("errors", []),
    }

    save_json("run_log.json", run_log)

    print(f"  Written to {output_dir}/")

    return {
        "output_files": [
            str(path) for path in output_dir.iterdir()
        ]
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_data", load_data)
    graph.add_node("generate_content", generate_content)
    graph.add_node("generate_images", generate_images)
    graph.add_node("competitor_analysis", competitor_analysis)
    graph.add_node("sales_analysis", sales_analysis)
    graph.add_node("write_output", write_output)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "generate_content")
    graph.add_edge("generate_content", "generate_images")
    graph.add_edge("generate_images", "competitor_analysis")
    graph.add_edge("competitor_analysis", "sales_analysis")
    graph.add_edge("sales_analysis", "write_output")
    graph.add_edge("write_output", END)

    return graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Agent - automated e-commerce pipeline"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to test_data.xlsx",
    )

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

    for error in errors:
        print(f"  - {error}")

    print("Output -> output/")
    print("Images -> images/")
