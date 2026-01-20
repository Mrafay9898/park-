import time
from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Utils --------------------
def timeit(fn: Callable):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        st.session_state.setdefault("timings", {})[fn.__name__] = round(elapsed, 4)
        return result
    return wrapper

# -------------------- Data Loader --------------------
@st.cache_data
@timeit
def load_data(path: str) -> pd.DataFrame:
    """Load and clean CSV data"""
    df = pd.read_csv(path, encoding="latin-1")
    
    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    
    # Convert price to numeric (handle errors)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Remove rows with missing prices for slider
    if "price" in df.columns:
        df = df.dropna(subset=["price"])
    
    return df

# -------------------- Filter Engine --------------------
class FilterEngine:
    def __init__(self, df: pd.DataFrame):
        self.base = df.copy()
        self.filters: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

    def clear(self):
        self.filters.clear()

    def add_filter(self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.filters[name] = func

    def apply(self) -> pd.DataFrame:
        df = self.base
        for fn in self.filters.values():
            df = fn(df)
        return df

# -------------------- Sidebar Filters --------------------
def build_sidebar(df: pd.DataFrame, engine: FilterEngine):
    st.sidebar.header("🔍 Filters")
    engine.clear()

    # Country filter
    if "country" in df.columns:
        countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
        country_sel = st.sidebar.selectbox("Country", countries)
        if country_sel != "All":
            engine.add_filter("country", lambda d: d[d["country"] == country_sel])

    # Category filter
    if "category" in df.columns:
        categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
        category_sel = st.sidebar.selectbox("Category", categories)
        if category_sel != "All":
            engine.add_filter("category", lambda d: d[d["category"] == category_sel])

    # Visitor reviews filter
    if "visitor_reviews" in df.columns:
        reviews = df["visitor_reviews"].dropna().unique().tolist()
        review_sel = st.sidebar.multiselect(
            "Visitor Reviews", 
            options=reviews, 
            default=reviews[:3] if len(reviews) > 3 else reviews
        )
        engine.add_filter("reviews", lambda d: d[d["visitor_reviews"].isin(review_sel)])

    # Price range filter
    if "price" in df.columns and not df["price"].isna().all():
        price_min = float(df["price"].min())
        price_max = float(df["price"].max())
        price_range = st.sidebar.slider(
            "Ticket Price Range ($)", 
            min_value=price_min, 
            max_value=price_max, 
            value=(price_min, price_max),
            step=0.1
        )
        engine.add_filter("price", lambda d: d[(d["price"] >= price_range[0]) & (d["price"] <= price_range[1])])

    # Show filtered count
    filtered_count = len(engine.apply())
    st.sidebar.markdown(f"**📊 Parks found:** {filtered_count}")
    st.sidebar.markdown(f"**📈 Total parks:** {len(df)}")

# -------------------- Main App --------------------
def main():
    st.set_page_config(page_title="Global Parks Dashboard", layout="wide")
    st.title("🌳 Global Parks Analytics Dashboard")
    st.caption("Filter parks by country, category, visitor reviews, and ticket price")

    # Load data
    csv_file = "Book1.csv"
    
    try:
        df = load_data(csv_file)
        st.success(f"✅ Loaded {len(df)} parks from {csv_file}")
        
        # Show data info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Parks", len(df))
        with col2:
            avg_price = df["price"].mean() if "price" in df.columns else 0
            st.metric("Avg Ticket Price", f"${avg_price:.2f}")

        # Build filters and apply
        engine = FilterEngine(df)
        build_sidebar(df, engine)
        filtered_df = engine.apply()

        # Show filtered results
        st.subheader("📋 Filtered Parks")
        if len(filtered_df) == 0:
            st.warning("❌ No parks match your filters. Try adjusting the filters.")
        else:
            st.dataframe(filtered_df, use_container_width=True)

        # Show timings
        if "timings" in st.session_state:
            st.sidebar.markdown("⏱️ **Performance**")
            for func, timing in st.session_state["timings"].items():
                st.sidebar.text(f"{func}: {timing}s")

    except FileNotFoundError:
        st.error(f"❌ File '{csv_file}' not found! Place it in the same folder as this script.")
        st.info("📁 Expected structure:\n- your_app.py\n- Book1.csv")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 Check if Book1.csv exists and has columns like 'country', 'category', 'visitor_reviews', 'price'")

if __name__ == "__main__":
    main()
