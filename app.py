import os
from dotenv import load_dotenv
load_dotenv()

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-3-1b-it")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMMA_LAST_ERROR = None

import re
import json
import sqlite3
from contextlib import contextmanager
from typing import Optional


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("SHAKTI_DB_PATH", os.path.join(APP_DIR, "shakti.db"))

app = FastAPI(title="Shakti", description="A local-first assistant for women running small vegetable businesses.")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS stock (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL UNIQUE COLLATE NOCASE,
            quantity REAL NOT NULL DEFAULT 0,
            unit TEXT NOT NULL DEFAULT 'kg'
        );
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT 'Marathi',
            notes TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            parsed_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT NOT NULL DEFAULT 'kg',
            FOREIGN KEY(order_id) REFERENCES orders(id)
        );
        """)
        count = con.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
        if count == 0:
            con.executemany("INSERT INTO stock(item, quantity, unit) VALUES (?, ?, ?)", [
                ("Tomato", 10, "kg"), ("Potato", 15, "kg"), ("Onion", 12, "kg"),
                ("Palak", 4, "bunch"), ("Bhendi", 5, "kg"), ("Coriander", 20, "bunch")
            ])


@app.on_event("startup")
def startup():
    init_db()


class StockUpdate(BaseModel):
    item: str = Field(min_length=1, max_length=80)
    quantity: float = Field(ge=0)
    unit: str = Field(default="kg", max_length=20)


class CustomerIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    phone: str = ""
    language: str = "Marathi"
    notes: str = ""


class OrderIn(BaseModel):
    customer_name: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=4000)


class MessageIn(BaseModel):
    customer_id: int
    language: Optional[str] = None


class ConfirmOrderIn(BaseModel):
    customer_name: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=4000)
    items: list[dict] = Field(min_length=1, max_length=30)


@app.get("/")
def home():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/api/stock")
def get_stock():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM stock ORDER BY item").fetchall()]


@app.post("/api/stock")
def set_stock(payload: StockUpdate):
    with db() as con:
        con.execute("""INSERT INTO stock(item, quantity, unit) VALUES (?, ?, ?)
            ON CONFLICT(item) DO UPDATE SET quantity=excluded.quantity, unit=excluded.unit""",
            (payload.item.strip(), payload.quantity, payload.unit.strip() or "kg"))
    return {"ok": True}


@app.get("/api/customers")
def get_customers():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM customers ORDER BY name").fetchall()]


@app.post("/api/customers")
def add_customer(payload: CustomerIn):
    with db() as con:
        cur = con.execute("INSERT INTO customers(name, phone, language, notes) VALUES (?, ?, ?, ?)",
                          (payload.name.strip(), payload.phone.strip(), payload.language, payload.notes.strip()))
        return dict(con.execute("SELECT * FROM customers WHERE id=?", (cur.lastrowid,)).fetchone())


def try_gemma(prompt: str) -> Optional[str]:
    """Run local Gemma on demand; fall back if disabled or unavailable."""
    global _generator, GEMMA_LAST_ERROR
    if os.getenv("SHAKTI_USE_GEMMA", "0") != "1":
        GEMMA_LAST_ERROR = "Gemma is disabled. Set SHAKTI_USE_GEMMA=1 and restart the server."
        return None
    try:
        from transformers import pipeline
        if "_generator" not in globals():
            kwargs = {"model": GEMMA_MODEL, "device_map": "auto"}
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN
            _generator = pipeline("text-generation", **kwargs)
        result = _generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=220, do_sample=False, return_full_text=False,
        )
        generated = result[0].get("generated_text", "")
        if isinstance(generated, list):
            for message in reversed(generated):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    answer = str(message.get("content", "")).strip()
                    if answer:
                        GEMMA_LAST_ERROR = None
                        return answer
            GEMMA_LAST_ERROR = "Gemma returned no assistant text."
            return None
        answer = str(generated).strip()
        GEMMA_LAST_ERROR = None if answer else "Gemma returned an empty response."
        return answer or None
    except Exception as exc:
        # Do not log prompts or customer data; preserve only a concise diagnostic.
        GEMMA_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[Shakti] Gemma unavailable ({type(exc).__name__}); using fallback.")
        return None


@app.get("/api/health")
def health_check():
    """Confirm which app is running and list supported order methods."""
    return {
        "ok": True,
        "app": "Shakti",
        "orders_methods": {"GET": "/api/orders", "POST": "/api/orders/preview then POST /api/orders"},
    }


@app.get("/api/gemma/status")
def gemma_status():
    """Report configuration and whether the model has loaded in this process."""
    try:
        import transformers  # noqa: F401
        transformers_installed = True
    except ImportError:
        transformers_installed = False
    loaded = "_generator" in globals()
    enabled = os.getenv("SHAKTI_USE_GEMMA", "0") == "1"
    return {
        "enabled": enabled,
        "model": GEMMA_MODEL,
        "transformers_installed": transformers_installed,
        "model_loaded_in_this_process": loaded,
        "last_error": GEMMA_LAST_ERROR,
        "hint": "Call POST /api/gemma/test to run an actual inference test." if enabled else "Set SHAKTI_USE_GEMMA=1 in PowerShell and restart Uvicorn.",
    }


@app.post("/api/gemma/test")
def gemma_test():
    """Actually run a small local inference, returning a clear success/failure result."""
    if os.getenv("SHAKTI_USE_GEMMA", "0") != "1":
        raise HTTPException(status_code=400, detail="Gemma is disabled. Set SHAKTI_USE_GEMMA=1 and restart Uvicorn.")
    answer = try_gemma("Reply with exactly this text and nothing else: SHAKTI_GEMMA_OK")
    if not answer:
        raise HTTPException(status_code=503, detail={"message": "Gemma inference failed; see /api/gemma/status and the terminal output.", "error": GEMMA_LAST_ERROR})
    return {"ok": True, "model": GEMMA_MODEL, "response": answer, "model_loaded_in_this_process": "_generator" in globals()}


ALIASES = {
    "टोमॅटो": "Tomato", "tomatoes": "Tomato", "tomato": "Tomato", "tamatar": "Tomato", "टमाटर": "Tomato",
    "बटाटा": "Potato", "बटाटे": "Potato", "potatoes": "Potato", "potato": "Potato", "batata": "Potato", "batate": "Potato", "aloo": "Potato", "आलू": "Potato",
    "कांदा": "Onion", "कांदे": "Onion", "onions": "Onion", "onion": "Onion", "kanda": "Onion", "प्याज": "Onion", "प्याज़": "Onion",
    "पालक": "Palak", "spinach": "Palak", "palak": "Palak", "bhaji": "Palak",
    "भेंडी": "Bhendi", "okra": "Bhendi", "bhendi": "Bhendi", "bhindi": "Bhendi", "भिंडी": "Bhendi",
    "कोथिंबीर": "Coriander", "coriander": "Coriander", "dhaniya": "Coriander", "धनिया": "Coriander", "धणे": "Coriander"
}
CANONICAL_ITEMS = sorted(set(ALIASES.values()))


def _normalise_model_items(parsed):
    """Validate model output against known item names and safe numeric quantities."""
    if not isinstance(parsed, list):
        return []
    cleaned = []
    for row in parsed[:30]:
        if not isinstance(row, dict):
            continue
        raw_item = str(row.get("item", "")).strip()
        canonical = ALIASES.get(raw_item.lower(), ALIASES.get(raw_item))
        # Permit canonical names, case-insensitively, but never accept arbitrary model inventions.
        if not canonical:
            canonical = next((x for x in CANONICAL_ITEMS if x.lower() == raw_item.lower()), None)
        try:
            qty = float(row.get("qty", row.get("quantity", 0)))
        except (TypeError, ValueError):
            continue
        if not canonical or not (0 < qty <= 500):
            continue
        default_unit = "bunch" if canonical in ("Palak", "Coriander") else "kg"
        unit_raw = str(row.get("unit", default_unit)).lower().strip()
        unit = "bunch" if unit_raw in ("bunch", "bunches", "जुडी", "जुडी", "जोडी") else "kg" if unit_raw in ("kg", "kilo", "kilos", "किलो", "किलो그램") else None
        if not unit:
            continue
        cleaned.append({"item": canonical, "qty": qty, "unit": unit})
    # Combine repeated item/unit lines to avoid duplicate stock calculations.
    combined = {}
    for item in cleaned:
        key = (item["item"], item["unit"])
        combined[key] = combined.get(key, 0) + item["qty"]
    return [{"item": k[0], "qty": v, "unit": k[1]} for k, v in combined.items()]


def parse_order(text: str):
    # Explicit cancellation language is not safely interpreted by the simple fallback.
    cancellation_markers = ("nako", "नको", "don't", "dont", "no ", "cancel")
    model_text = try_gemma(
        "You extract vegetable purchase orders for a small Indian vegetable seller.\n"
        "The customer may mix Marathi, Hindi, and English, use transliteration, abbreviations, or corrections.\n"
        f"Recognised item catalogue and canonical output names: {', '.join(CANONICAL_ITEMS)}.\n"
        "Return ONLY a JSON object with keys items and needs_clarification. Example: "
        '{"items":[{"item":"Tomato","qty":2,"unit":"kg"}],"needs_clarification":false}.\n'
        "Use only catalogue items. Include only items the customer wants to buy, not items they reject. "
        "Do not guess missing quantities; if a requested item's quantity is unclear, set needs_clarification=true. "
        "Allowed units are kg and bunch. Convert known aliases to canonical names.\n"
        "Customer message: " + text
    )
    if model_text:
        try:
            start, end = model_text.index("{"), model_text.rindex("}") + 1
            payload = json.loads(model_text[start:end])
            cleaned = _normalise_model_items(payload.get("items", []))
            if payload.get("needs_clarification") or not cleaned:
                return [], "gemma", True
            return cleaned, "gemma", False
        except Exception:
            pass

    # Convert a starter vocabulary of Marathi/Hindi/transliterated number words.
    # This is intentionally limited; the seller still reviews every parsed order.
    normalised_text = text.lower()
    number_words = {
        "एक": "1", "दोन": "2", "तीन": "3", "चार": "4", "पाच": "5",
        "सहा": "6", "सात": "7", "आठ": "8", "नऊ": "9", "दहा": "10",
        "दो": "2", "पाँच": "5", "पांच": "5", "छह": "6", "नौ": "9", "दस": "10",
        "ek": "1", "don": "2", "doun": "2", "teen": "3", "char": "4", "chaar": "4",
        "paach": "5", "pach": "5", "saha": "6", "saat": "7", "aath": "8", "nau": "9",
        "daha": "10", "do": "2", "paanch": "5", "das": "10"
    }
    for word, digit in sorted(number_words.items(), key=lambda pair: len(pair[0]), reverse=True):
        if word.isascii():
            normalised_text = re.sub(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", digit, normalised_text)
        else:
            normalised_text = normalised_text.replace(word, digit)

    # Parse clauses separately so a rejection such as "kanda nako, 1 kg tomato"
    # does not accidentally reject the tomato too.
    cleaned = []
    clauses = re.split(r"[,;\n]|\band\b|\bani\b|\baur\b|\baani\b| आणि | और | व ", normalised_text)
    for clause in clauses:
        if any(marker in clause for marker in cancellation_markers):
            continue
        for alias in sorted(ALIASES.keys(), key=len, reverse=True):
            alias_lower = alias.lower()
            qty_pattern = r"(\d+(?:\.\d+)?)\s*(kg|kilo|किलो|bunch|जुडी|जोडी)?\s*" + re.escape(alias_lower)
            match = re.search(qty_pattern, clause)
            if not match:
                qty_pattern = re.escape(alias_lower) + r"\s*(\d+(?:\.\d+)?)\s*(kg|kilo|किलो|bunch|जुडी|जोडी)?"
                match = re.search(qty_pattern, clause)
            if not match:
                continue
            qty = float(match.group(1))
            unit = match.group(2)
            item = ALIASES[alias]
            unit = (unit or ("bunch" if item in ("Palak", "Coriander") else "kg")).lower()
            unit = "bunch" if unit in ("bunch", "जुडी", "जोडी") else "kg"
            if qty <= 0 or qty > 500:
                continue
            key = (item, unit)
            existing = next((x for x in cleaned if (x["item"], x["unit"]) == key), None)
            if existing:
                existing["qty"] += qty
            else:
                cleaned.append({"item": item, "qty": qty, "unit": unit})
    return cleaned, "regex", False


@app.post("/api/orders/preview")
def preview_order(payload: OrderIn):
    """Parse an order without changing stock or saving it; the seller must confirm."""
    items, parser, needs_clarification = parse_order(payload.text)
    if not items:
        raise HTTPException(status_code=422, detail=(
            "I couldn't safely identify the order. Try '2 kg tomato, 1 kg potato, 1 bunch palak' "
            "or clarify the quantity/item before recording it."
        ))
    with db() as con:
        stock = {r["item"].lower(): dict(r) for r in con.execute("SELECT * FROM stock").fetchall()}
    annotated = []
    for item in items:
        current = stock.get(item["item"].lower())
        annotated.append({**item, "stock_available": current["quantity"] if current else None,
                          "stock_unit": current["unit"] if current else None,
                          "warning": "Item is not in stock list" if current is None else
                          ("Order exceeds available stock" if current["unit"] == item["unit"] and item["qty"] > current["quantity"] else None)})
    return {"customer_name": payload.customer_name, "text": payload.text, "items": annotated,
            "parser": parser, "needs_clarification": needs_clarification, "requires_confirmation": True}


@app.post("/api/orders")
def create_order(payload: ConfirmOrderIn):
    """Save only seller-reviewed items and then update inventory."""
    items = _normalise_model_items(payload.items)
    if not items:
        raise HTTPException(status_code=422, detail="No valid order items to record.")
    with db() as con:
        cur = con.execute("INSERT INTO orders(customer_name, raw_text, parsed_json) VALUES (?, ?, ?)",
                          (payload.customer_name.strip(), payload.text.strip(), json.dumps(items, ensure_ascii=False)))
        order_id = cur.lastrowid
        for item in items:
            con.execute("INSERT INTO order_items(order_id, item, quantity, unit) VALUES (?, ?, ?, ?)",
                        (order_id, item["item"], item["qty"], item["unit"]))
            stock = con.execute("SELECT quantity, unit FROM stock WHERE item=? COLLATE NOCASE", (item["item"],)).fetchone()
            if stock and stock["unit"] == item["unit"]:
                con.execute("UPDATE stock SET quantity=? WHERE item=? COLLATE NOCASE",
                            (max(0, float(stock["quantity"]) - item["qty"]), item["item"]))
        return {"id": order_id, "customer_name": payload.customer_name, "items": items, "parser": "seller-confirmed"}


@app.get("/api/orders")
def get_orders():
    """Return recent orders for the dashboard without exposing raw customer messages elsewhere."""
    with db() as con:
        rows = con.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 30").fetchall()
        return [dict(row) | {"items": json.loads(row["parsed_json"])} for row in rows]


def customer_favourites(con, customer_name: str):
    rows = con.execute("""
        SELECT oi.item, SUM(oi.quantity) AS total
        FROM order_items oi JOIN orders o ON o.id=oi.order_id
        WHERE lower(o.customer_name)=lower(?)
        GROUP BY oi.item ORDER BY total DESC LIMIT 3
    """, (customer_name,)).fetchall()
    return [r["item"] for r in rows]


def make_message(customer, stock):
    language = customer["language"]
    favourites = customer["favourites"]
    available = [s["item"] for s in stock if s["quantity"] > 0]
    usual = [x for x in favourites if x in available]
    new_item = next((x for x in available if x not in favourites), None)
    context = {
        "customer_name": customer["name"],
        "language": language,
        "usual_items_in_stock": usual[:2],
        "one_optional_new_item_in_stock": new_item,
        "all_items_currently_in_stock": available,
    }
    prompt = (
        "Write a warm, short WhatsApp message for a customer of a small Indian vegetable seller. "
        "Write only in the requested language: " + language + ". "
        "Mention the customer's usual items first when present, then suggest at most one new item. "
        "Use only vegetable names supplied in the context. Never invent prices, discounts, quantities, "
        "delivery promises, freshness claims, or stock. If no usual item is listed, use a general greeting. "
        "Keep it natural and under 45 words. Return only the message, with no explanation. Context: "
        + json.dumps(context, ensure_ascii=False)
    )
    generated = try_gemma(prompt)
    if generated:
        return generated.strip().strip('"')
    if language == "Hindi":
        usual_text = ", ".join(usual) if usual else "आज की ताज़ी सब्ज़ियाँ"
        extra = f" आज {new_item} भी ताज़ा है—चाहें तो जोड़ दूँ।" if new_item else ""
        return f"नमस्ते {customer['name']} जी! आपके पसंदीदा {usual_text} आज ताज़ा हैं।{extra} बताइए, क्या भेजूँ?"
    if language == "English":
        usual_text = ", ".join(usual) if usual else "today's fresh vegetables"
        extra = f" You could also try today's fresh {new_item}." if new_item else ""
        return f"Hi {customer['name']}! Your usual {usual_text} are fresh today.{extra} Would you like me to keep some aside?"
    usual_text = ", ".join(usual) if usual else "आजच्या ताज्या भाज्या"
    extra = f" आजची ताजी {new_item} पण घेऊन बघा." if new_item else ""
    return f"नमस्कार {customer['name']}! तुमची नेहमीची {usual_text} आज ताजी आहे.{extra} तुमच्यासाठी ठेवू का?"


@app.get("/api/messages")
def messages():
    with db() as con:
        customers = [dict(r) for r in con.execute("SELECT * FROM customers ORDER BY name").fetchall()]
        stock = [dict(r) for r in con.execute("SELECT * FROM stock ORDER BY item").fetchall()]
        result = []
        for customer in customers:
            customer["favourites"] = customer_favourites(con, customer["name"])
            customer["message"] = make_message(customer, stock)
            result.append(customer)
        return result


@app.post("/api/messages")
def one_message(payload: MessageIn):
    with db() as con:
        row = con.execute("SELECT * FROM customers WHERE id=?", (payload.customer_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Customer not found")
        customer = dict(row)
        if payload.language:
            customer["language"] = payload.language
        customer["favourites"] = customer_favourites(con, customer["name"])
        stock = [dict(r) for r in con.execute("SELECT * FROM stock ORDER BY item").fetchall()]
        return {"customer": customer, "message": make_message(customer, stock)}


@app.get("/api/summary")
def summary():
    with db() as con:
        return {
            "customers": con.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
            "orders": con.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
            "stock_items": con.execute("SELECT COUNT(*) FROM stock").fetchone()[0],
            "available_items": con.execute("SELECT COUNT(*) FROM stock WHERE quantity > 0").fetchone()[0]
        }


@app.get("/api/menu-post")
def menu_post():
    with db() as con:
        stock = [dict(r) for r in con.execute("SELECT * FROM stock WHERE quantity > 0 ORDER BY item").fetchall()]
    items = ", ".join(f"{s['item']} ({s['quantity']:g} {s['unit']})" for s in stock)
    return {"message": f"🌿 आजचा ताजा भाजीपाला / Today's fresh harvest: {items}. Message me to reserve your vegetables! 🥬"}
