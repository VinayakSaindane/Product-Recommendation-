import json
import os
import re
from html import escape
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # The app still works with the local NLP fallback.
    genai = None
    genai_types = None


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RUPEE = "Rs."

PRODUCT_SYNONYMS = {
    "lipstick": (
        "lipstick",
        "lipsticks",
        "lip color",
        "lip colour",
        "lip crayon",
        "lip stain",
        "lip gloss",
    ),
    "lip balm": ("lip balm", "lipbalm", "balm"),
    "shoe": ("shoe", "shoes", "sneaker", "sneakers", "loafers", "lace up", "slip on", "footwear", "boots"),
    "shirt": ("shirt", "shirts"),
    "t-shirt": ("t-shirt", "tshirts", "tee", "round neck"),
    "dress": ("dress", "dresses", "gown"),
    "watch": ("watch", "watches"),
    "bag": ("bag", "backpack", "handbag", "wallet", "pouch"),
    "sofa": ("sofa", "sofa bed"),
    "shorts": ("shorts",),
    "sandal": ("sandal", "sandals", "wedges", "heels"),
    "nail polish": ("nail polish", "nail paint"),
    "kurti": ("kurti", "kurtis"),
    "bra": ("bra", "bras"),
    "necklace": ("necklace", "necklaces", "chain", "chains"),
}

PRODUCT_CATEGORY_HINTS = {
    "lipstick": ("lipsticks", "lipstick", "lip color", "lip colours"),
    "lip balm": ("lip balms", "lip balm", "lip care"),
    "shoe": ("shoes", "casual shoes", "formal shoes", "sports shoes", "loafers", "boots"),
    "shirt": ("clothing", "shirts"),
    "t-shirt": ("clothing", "t-shirts", "polos"),
    "dress": ("clothing", "dresses", "gowns"),
    "watch": ("watches",),
    "bag": ("bags", "wallets"),
    "sofa": ("furniture", "sofa"),
    "shorts": ("clothing", "shorts"),
    "sandal": ("footwear", "sandals", "wedges", "heels"),
    "nail polish": ("nail polishes", "nail polish", "nail paint"),
    "kurti": ("clothing", "ethnic wear", "kurtis"),
    "bra": ("clothing", "lingerie", "bras"),
    "necklace": ("jewellery", "necklaces", "chains"),
}

COLOR_TERMS = {
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "gray",
    "gold",
    "maroon",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}

OCCASION_TERMS = ("formal", "casual", "party", "ethnic", "sports", "office", "school", "wedding")
ATTRIBUTE_TERMS = (
    "comfortable",
    "leather",
    "cotton",
    "linen",
    "slim",
    "solid",
    "printed",
    "waterproof",
    "matte",
    "glossy",
    "cream",
    "waterproof",
)


@dataclass
class QueryIntent:
    raw_query: str
    product_type: str = ""
    occasion: str = ""
    gender: str = ""
    max_price: float | None = None
    budget_intent: bool = False
    preferred_brands: tuple[str, ...] = ()
    category_terms: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    sort: str = "relevance"

    @property
    def search_phrase(self) -> str:
        parts = [
            self.raw_query,
            self.product_type,
            self.occasion,
            self.gender,
            " ".join(self.category_terms),
            " ".join(self.attributes),
            " ".join(self.colors),
            " ".join(self.preferred_brands),
        ]
        return " ".join(part for part in parts if part).strip()


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalized_tokens(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    tokens: list[str] = []
    for value in values:
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
        tokens.extend(token for token in cleaned.split() if len(token) > 2)
    return tuple(dict.fromkeys(tokens))


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    return any(f" {re.sub(r'[^a-z0-9]+', ' ', phrase.lower()).strip()} " in normalized for phrase in phrases)


def _canonical_product_type(*values: str) -> str:
    text = " ".join(str(value).lower() for value in values if value)
    for canonical, variants in PRODUCT_SYNONYMS.items():
        if canonical in text or _contains_phrase(text, variants):
            return canonical
    return ""


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _price_from_query(query: str) -> float | None:
    query = query.lower().replace(",", "")
    patterns = [
        r"(?:under|below|less than|up to|max(?:imum)?|within)\s*(?:rs\.?|inr|\u20b9)?\s*(\d{2,7})",
        r"(?:rs\.?|inr|\u20b9)\s*(\d{2,7})",
        r"(\d{2,7})\s*(?:rs|inr|rupees)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return float(match.group(1))
    return None


def _fallback_intent(query: str) -> QueryIntent:
    text = query.lower()

    product_type = _canonical_product_type(text)

    gender = ""
    if re.search(r"\b(men|mens|male|boys|boy)\b", text):
        gender = "Men"
    elif re.search(r"\b(women|womens|female|girls|girl|ladies)\b", text):
        gender = "Women"

    budget_intent = any(word in text for word in ("budget", "cheap", "affordable", "low price", "value", "economical"))
    occasion = next((term for term in OCCASION_TERMS if term in text), "")
    colors = tuple(term for term in COLOR_TERMS if _contains_phrase(text, (term,)))
    attributes = tuple(term for term in ATTRIBUTE_TERMS if _contains_phrase(text, (term,)))

    return QueryIntent(
        raw_query=query,
        product_type=product_type,
        occasion=occasion,
        gender=gender,
        max_price=_price_from_query(query),
        budget_intent=budget_intent,
        category_terms=tuple(filter(None, (product_type, occasion))),
        attributes=attributes,
        colors=colors,
        sort="price_low" if budget_intent else "relevance",
    )


@st.cache_data(show_spinner=False)
def _gemini_intent(query: str, api_key: str) -> dict[str, Any]:
    if not api_key or genai is None:
        return {}

    prompt = f"""
Extract shopping intent from this customer query as compact JSON only.

Query: {query}

Schema:
{{
  "product_type": "main product noun, singular if possible",
  "occasion": "formal/casual/sports/party/office/school/ethnic/wedding or empty",
  "gender": "Men/Women/Unisex or empty",
  "max_price": number or null,
  "budget_intent": true/false,
  "preferred_brands": ["brand names explicitly requested"],
  "category_terms": ["important category words"],
  "attributes": ["materials, style, use case, comfort, pattern, fit"],
  "colors": ["requested colors"],
  "sort": "price_low/relevance/discount"
}}
Use Indian rupee amounts when the user gives prices. Do not invent brands.
"""
    try:
        client = genai.Client(api_key=api_key)
        config = None
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            )
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
        return _extract_json(response.text or "")
    except Exception:
        return {}


def parse_query_intent(query: str) -> QueryIntent:
    fallback = _fallback_intent(query)
    api_key = os.getenv("GEMINI_API_KEY", "")
    parsed = _gemini_intent(query, api_key)
    if not parsed:
        return fallback

    def pick(name: str, default: Any) -> Any:
        value = parsed.get(name)
        return default if value in (None, "", []) else value

    max_price = pick("max_price", fallback.max_price)
    try:
        max_price = float(max_price) if max_price is not None else None
    except (TypeError, ValueError):
        max_price = fallback.max_price

    product_type = _canonical_product_type(
        str(pick("product_type", fallback.product_type)),
        " ".join(map(str, pick("category_terms", fallback.category_terms))),
        " ".join(map(str, pick("attributes", fallback.attributes))),
    ) or fallback.product_type

    return QueryIntent(
        raw_query=query,
        product_type=product_type,
        occasion=str(pick("occasion", fallback.occasion)).strip(),
        gender=str(pick("gender", fallback.gender)).strip().title(),
        max_price=max_price,
        budget_intent=bool(pick("budget_intent", fallback.budget_intent)),
        preferred_brands=tuple(pick("preferred_brands", fallback.preferred_brands)),
        category_terms=tuple(pick("category_terms", fallback.category_terms)),
        attributes=tuple(pick("attributes", fallback.attributes)),
        colors=tuple(pick("colors", fallback.colors)),
        sort=str(pick("sort", fallback.sort)).strip() or fallback.sort,
    )


@st.cache_resource(show_spinner=False)
def build_search_index(refined_df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    catalog_text = refined_df["search_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=18000)
    matrix = vectorizer.fit_transform(catalog_text)
    return vectorizer, matrix


def _prepare_catalog(refined_df: pd.DataFrame) -> pd.DataFrame:
    catalog = refined_df.copy()
    for column in ["brand", "description", "product_name", "primary_category", "category_path", "gender"]:
        catalog[column] = catalog[column].fillna("").map(_clean_text)

    catalog["retail_price"] = pd.to_numeric(catalog["retail_price"], errors="coerce")
    catalog["discounted_price"] = pd.to_numeric(catalog["discounted_price"], errors="coerce")
    catalog = catalog.dropna(subset=["discounted_price", "retail_price"])
    catalog["discount_pct"] = np.where(
        catalog["retail_price"] > 0,
        ((catalog["retail_price"] - catalog["discounted_price"]) / catalog["retail_price"] * 100).clip(lower=0),
        0,
    )
    catalog["search_text"] = (
        catalog["product_name"]
        + " "
        + catalog["brand"]
        + " "
        + catalog["primary_category"]
        + " "
        + catalog["category_path"]
        + " "
        + catalog["gender"]
        + " "
        + catalog["description"].str.slice(0, 900)
    ).str.lower()
    catalog = catalog.drop_duplicates(subset=["product_name", "discounted_price", "brand"])
    return catalog.reset_index(drop=True)


def _product_type_mask(catalog: pd.DataFrame, product_type: str) -> pd.Series:
    if not product_type:
        return pd.Series(False, index=catalog.index)

    synonyms = PRODUCT_SYNONYMS.get(product_type, (product_type,))
    category_hints = PRODUCT_CATEGORY_HINTS.get(product_type, ())
    name_brand = (catalog["product_name"] + " " + catalog["brand"]).str.lower()
    category_path = catalog["category_path"].str.lower()

    name_match = pd.Series(False, index=catalog.index)
    for phrase in synonyms:
        name_match = name_match | name_brand.str.contains(rf"\b{re.escape(phrase.lower())}\b", regex=True)

    category_match = pd.Series(False, index=catalog.index)
    for phrase in (*synonyms, *category_hints):
        category_match = category_match | category_path.str.contains(re.escape(phrase.lower()), regex=True)

    return name_match | category_match


def _term_mask(catalog: pd.DataFrame, terms: tuple[str, ...]) -> pd.Series:
    if not terms:
        return pd.Series(False, index=catalog.index)
    mask = pd.Series(False, index=catalog.index)
    for term in terms:
        mask = mask | catalog["search_text"].str.contains(rf"\b{re.escape(term.lower())}\b", regex=True)
    return mask


def recommend_products(refined_df: pd.DataFrame, query: str, limit: int = 8) -> tuple[QueryIntent, pd.DataFrame]:
    catalog = _prepare_catalog(refined_df)
    intent = parse_query_intent(query)
    vectorizer, matrix = build_search_index(catalog)
    query_vector = vectorizer.transform([intent.search_phrase])
    semantic_score = cosine_similarity(query_vector, matrix).ravel()

    price = catalog["discounted_price"]
    price_score = 1 - ((price - price.min()) / max(price.max() - price.min(), 1))
    discount_score = catalog["discount_pct"] / 100
    score = semantic_score * 0.6 + price_score.to_numpy() * 0.16 + discount_score.to_numpy() * 0.12

    text = catalog["search_text"]
    product_match = _product_type_mask(catalog, intent.product_type)
    if intent.product_type:
        score += product_match.to_numpy() * 0.34

    core_terms = _normalized_tokens([intent.occasion, *intent.category_terms, *intent.attributes])
    for term in core_terms:
        if term not in COLOR_TERMS and term != intent.product_type:
            score += text.str.contains(rf"\b{re.escape(term)}\b", regex=True).to_numpy() * 0.035

    for term in _normalized_tokens(list(intent.colors)):
        score += text.str.contains(rf"\b{re.escape(term)}\b", regex=True).to_numpy() * 0.04

    if intent.gender in {"Men", "Women"}:
        gender_match = (catalog["gender"].eq(intent.gender) | text.str.contains(intent.gender.lower())).to_numpy()
        score += gender_match * 0.07

    for brand in intent.preferred_brands:
        score += catalog["brand"].str.lower().str.contains(re.escape(str(brand).lower()), regex=True).to_numpy() * 0.12

    candidates = catalog.copy()
    candidates["match_score"] = score

    if intent.product_type:
        product_candidates = candidates[product_match]
        if len(product_candidates) >= 2:
            candidates = product_candidates

    if intent.max_price:
        within_budget = candidates["discounted_price"] <= intent.max_price
        if within_budget.sum() >= 3:
            candidates = candidates[within_budget]

    if intent.budget_intent and not intent.max_price:
        ceiling = candidates["discounted_price"].quantile(0.55)
        value_candidates = candidates[candidates["discounted_price"] <= ceiling]
        if len(value_candidates) >= 5:
            candidates = value_candidates

    if intent.gender in {"Men", "Women"}:
        gender_candidates = candidates[candidates["gender"].isin([intent.gender, "Unisex"])]
        if len(gender_candidates) >= 3:
            candidates = gender_candidates

    strict_terms = [term for term in _normalized_tokens([intent.occasion]) if term]
    for term in strict_terms:
        term_matches = candidates["search_text"].str.contains(re.escape(term), regex=True)
        if term_matches.sum() >= 3:
            candidates = candidates[term_matches]

    candidates = candidates.copy()
    color_matches = _term_mask(candidates, intent.colors)
    if color_matches.sum() >= 1:
        candidates.loc[color_matches, "match_score"] += 0.22

    descriptive_matches = _term_mask(candidates, intent.attributes)
    if descriptive_matches.sum() >= 1:
        candidates.loc[descriptive_matches, "match_score"] += 0.08

    attribute_matches = _term_mask(candidates, tuple(term for term in (*intent.colors, *intent.attributes) if term))
    if attribute_matches.sum() >= 3:
        candidates = candidates[attribute_matches | candidates.index.isin(candidates.nlargest(3, "match_score").index)]

    if intent.sort == "price_low" or intent.budget_intent:
        candidates = candidates.sort_values(["match_score", "discounted_price"], ascending=[False, True])
    elif intent.sort == "discount":
        candidates = candidates.sort_values(["match_score", "discount_pct"], ascending=[False, False])
    else:
        candidates = candidates.sort_values(["match_score", "discounted_price"], ascending=[False, True])

    return intent, candidates.head(limit).reset_index(drop=True)


def _format_price(value: float) -> str:
    return f"{RUPEE} {value:,.0f}"


def _render_intent_chips(intent: QueryIntent) -> None:
    chips = []
    if intent.product_type:
        chips.append(intent.product_type.title())
    if intent.occasion:
        chips.append(intent.occasion.title())
    if intent.gender:
        chips.append(intent.gender)
    if intent.max_price:
        chips.append(f"Under {_format_price(intent.max_price)}")
    if intent.budget_intent:
        chips.append("Budget friendly")
    chips.extend(str(color).title() for color in intent.colors[:3])

    if chips:
        st.markdown(
            '<div class="chip-row">' + "".join(f"<span>{chip}</span>" for chip in chips) + "</div>",
            unsafe_allow_html=True,
        )


def _render_product_card(product: pd.Series, rank: int) -> None:
    price = _format_price(product["discounted_price"])
    retail = _format_price(product["retail_price"])
    discount = int(round(product.get("discount_pct", 0)))
    image = product.get("primary_image_link") or ""
    name = escape(str(product["product_name"]))
    brand = escape(str(product.get("brand") or "Brand not listed"))
    category = escape(str(product.get("primary_category") or "Product"))
    gender = escape(str(product.get("gender", "Unisex")))
    description = escape(_clean_text(product.get("description"))[:190])
    score = min(max(float(product.get("match_score", 0)) * 100, 0), 100)

    image_html = (
        f'<img src="{escape(str(image))}" alt="{name}">'
        if image.startswith("http")
        else '<div class="image-fallback">No Image</div>'
    )
    link = escape(str(product.get("product_url") or "#"))

    st.markdown(
        f"""
        <article class="product-card">
            <div class="product-image">{image_html}</div>
            <div class="product-copy">
                <div class="card-topline">
                    <span>#{rank} match</span>
                    <span>{score:.0f}% fit</span>
                </div>
                <h3>{name}</h3>
                <p class="meta">{brand} &middot; {category} &middot; {gender}</p>
                <p class="desc">{description}</p>
                <div class="price-row">
                    <strong>{price}</strong>
                    <s>{retail}</s>
                    <span>{discount}% off</span>
                </div>
                <a class="product-link" href="{link}" target="_blank">View product</a>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def display_product_recommendation(refined_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <section class="recommend-hero">
            <div>
                <p class="eyebrow">Gemini shopping intelligence</p>
                <h1>Ask like a real customer. Get ranked products from the catalog.</h1>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        "find me budget friendly shoes that are formal",
        "black office footwear for men under 1000",
        "comfortable cotton shorts for women at a low price",
        "party wear dress with good discount",
    ]

    selected = st.pills("Try a query", examples, selection_mode="single")
    default_query = selected or examples[0]
    query = st.text_area(
        "Shopping query",
        value=default_query,
        height=92,
        placeholder="Example: find me budget friendly shoes that are formal",
    )

    cols = st.columns([1, 1, 2])
    with cols[0]:
        result_count = st.slider("Results", min_value=3, max_value=12, value=6)
    with cols[1]:
        run_search = st.button("Recommend", type="primary", use_container_width=True)

    if not run_search and "last_query" not in st.session_state:
        run_search = True

    if run_search:
        st.session_state["last_query"] = query

    active_query = st.session_state.get("last_query", query).strip()
    if not active_query:
        st.info("Type what you want to buy.")
        return

    with st.spinner("Understanding the query and ranking products..."):
        intent, products = recommend_products(refined_df, active_query, result_count)

    _render_intent_chips(intent)

    if products.empty:
        st.warning("No strong catalog matches found. Try widening the price, brand, or style.")
        return

    st.markdown(f"### Best matches for \"{active_query}\"")
    for index, product in products.iterrows():
        _render_product_card(product, index + 1)
