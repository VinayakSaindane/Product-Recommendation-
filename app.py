import streamlit as st

from data_processing import display_data_analysis, load_data, preprocess_data
from recommendation import display_product_recommendation


def apply_theme():
    st.markdown(
        """
        <style>
            :root {
                --ink: #121826;
                --muted: #526172;
                --line: #d7dee8;
                --paper: #ffffff;
                --wash: #f7f4ee;
                --panel: #fbfaf7;
                --accent: #0e7c73;
                --accent-deep: #094d4a;
                --gold: #bd8b13;
                --coral: #c94f3d;
                --shadow: 0 18px 48px rgba(18, 24, 38, 0.1);
            }

            .stApp {
                background:
                    radial-gradient(circle at 12% 6%, rgba(14, 124, 115, 0.12), transparent 28%),
                    radial-gradient(circle at 92% 12%, rgba(189, 139, 19, 0.12), transparent 24%),
                    linear-gradient(135deg, #fbfaf7 0%, #f5f7fb 44%, #fff7f2 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1180px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(18, 24, 38, 0.98) 0%, rgba(16, 58, 62, 0.98) 60%, rgba(9, 77, 74, 0.98) 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.16);
            }

            [data-testid="stSidebar"] * {
                color: #f9fafb !important;
            }

            [data-testid="stSidebar"] [data-baseweb="radio"] label,
            [data-testid="stSidebar"] .stCaption {
                color: #e8eef5 !important;
            }

            .sidebar-brand {
                border: 1px solid rgba(255, 255, 255, 0.22);
                background: linear-gradient(145deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.06));
                border-radius: 8px;
                padding: 1.05rem;
                margin: 0.35rem 0 1rem;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
            }

            .sidebar-brand h2 {
                margin: 0;
                color: #ffffff;
                font-size: 1.45rem;
                line-height: 1.05;
            }

            .sidebar-brand p {
                margin: 0.45rem 0 0;
                color: #dbeafe !important;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            [data-testid="stSidebar"] [role="radiogroup"] {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 8px;
                padding: 0.45rem;
            }

            [data-testid="stSidebar"] [role="radio"] {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 0.35rem 0.45rem;
                margin: 0.2rem 0;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            h1, h2, h3, h4,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stWidgetLabel"] p,
            label {
                color: var(--ink);
            }

            [data-testid="stWidgetLabel"] {
                margin-bottom: 0.3rem;
            }

            [data-testid="stTextArea"],
            [data-testid="stSlider"],
            [data-testid="stButton"] {
                position: relative;
                z-index: 2;
            }

            textarea, input,
            [data-baseweb="textarea"] textarea,
            [data-baseweb="input"] input {
                background: #ffffff !important;
                color: #111827 !important;
                border: 1px solid #b6c2cf !important;
                border-radius: 8px !important;
                box-shadow: 0 10px 26px rgba(18, 24, 38, 0.07);
            }

            textarea::placeholder, input::placeholder {
                color: #64748b !important;
                opacity: 1 !important;
            }

            [data-baseweb="tag"],
            [data-baseweb="tag"] span,
            button[data-baseweb="tag"],
            button[data-baseweb="tag"] span {
                color: #122033 !important;
                background: #ffffff !important;
                border-color: #cbd5e1 !important;
                font-weight: 700;
            }

            div[data-testid="stPills"] button,
            div[data-testid="stPills"] button span,
            div[data-testid="stPills"] button p {
                color: #122033 !important;
                background: #ffffff !important;
                border-color: #cbd5e1 !important;
                white-space: normal !important;
                line-height: 1.25 !important;
            }

            div[data-testid="stPills"] button[aria-pressed="true"],
            div[data-testid="stPills"] button[aria-selected="true"] {
                background: #e8f6f2 !important;
                border-color: #0e7c73 !important;
            }

            div[data-testid="stPills"] {
                margin-bottom: 0.65rem;
            }

            .app-shell {
                padding: 1.4rem 0 1.7rem;
            }

            .app-title {
                font-size: clamp(2rem, 4vw, 4.2rem);
                line-height: 1;
                font-weight: 850;
                max-width: 920px;
                margin: 0;
                color: var(--ink);
                text-wrap: balance;
            }

            .app-subtitle {
                max-width: 720px;
                color: var(--muted);
                font-size: 1.08rem;
                margin-top: 0.8rem;
                line-height: 1.62;
            }

            .metric-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 1.1rem 0 0.4rem;
            }

            .metric-tile {
                border: 1px solid rgba(18, 24, 38, 0.1);
                background: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                padding: 0.95rem 1rem;
                box-shadow: 0 12px 32px rgba(18, 24, 38, 0.07);
            }

            .metric-tile strong {
                display: block;
                font-size: 1.45rem;
                line-height: 1.1;
            }

            .metric-tile span {
                color: var(--muted);
                font-size: 0.86rem;
            }

            .recommend-hero {
                border: 1px solid rgba(18, 24, 38, 0.1);
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.93), rgba(247, 244, 238, 0.9));
                border-radius: 8px;
                box-shadow: var(--shadow);
                margin-bottom: 1.25rem;
                padding: 1.15rem 1.2rem 1.25rem;
            }

            .recommend-hero h1 {
                font-size: clamp(1.8rem, 3vw, 3rem);
                line-height: 1.05;
                margin: 0;
                max-width: 820px;
                color: var(--ink);
                text-wrap: balance;
            }

            .eyebrow {
                margin: 0 0 0.35rem;
                color: var(--gold) !important;
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin: 0.8rem 0 1rem;
            }

            .chip-row span {
                background: #effaf7;
                color: #094d4a !important;
                border: 1px solid #b7dfd7;
                border-radius: 999px;
                padding: 0.35rem 0.7rem;
                font-size: 0.86rem;
                font-weight: 700;
                line-height: 1.2;
            }

            .product-card {
                display: grid;
                grid-template-columns: 176px minmax(0, 1fr);
                gap: 1rem;
                align-items: stretch;
                margin: 0.95rem 0;
                border: 1px solid rgba(15, 23, 42, 0.14);
                background: linear-gradient(145deg, #ffffff, #fbfaf7);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: var(--shadow);
            }

            .product-image {
                min-height: 184px;
                background: linear-gradient(135deg, #f8fafc, #f1f5f9);
                display: grid;
                place-items: center;
                border-right: 1px solid rgba(23, 32, 42, 0.08);
            }

            .product-image img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                padding: 0.7rem;
            }

            .image-fallback {
                color: var(--muted);
                font-weight: 700;
            }

            .product-copy {
                min-width: 0;
                padding: 1rem 1.1rem 1rem 0;
            }

            .card-topline {
                display: flex;
                justify-content: space-between;
                gap: 0.75rem;
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
            }

            .card-topline span {
                color: var(--accent-deep) !important;
                line-height: 1.2;
            }

            .product-card h3 {
                margin: 0.35rem 0 0.2rem;
                font-size: 1.15rem;
                line-height: 1.28;
                color: var(--ink);
                overflow-wrap: anywhere;
            }

            .meta, .desc {
                margin: 0.25rem 0;
                color: #465568 !important;
                line-height: 1.45;
                overflow-wrap: anywhere;
            }

            .desc {
                min-height: auto;
                max-width: 760px;
            }

            .price-row {
                display: flex;
                flex-wrap: wrap;
                align-items: baseline;
                gap: 0.55rem;
                margin: 0.65rem 0;
            }

            .price-row strong {
                font-size: 1.25rem;
                color: var(--ink) !important;
            }

            .price-row s {
                color: #728197 !important;
            }

            .price-row span {
                color: #0b7a46 !important;
                font-weight: 800;
            }

            .product-link {
                color: #ffffff !important;
                background: linear-gradient(135deg, var(--ink), var(--accent-deep));
                border-radius: 8px;
                padding: 0.5rem 0.8rem;
                text-decoration: none !important;
                display: inline-flex;
                font-weight: 800;
                line-height: 1.2;
            }

            div[data-testid="stButton"] button {
                border-radius: 8px;
                font-weight: 800;
            }

            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(135deg, #0e7c73, #094d4a) !important;
                border-color: #0e7c73 !important;
                color: #ffffff !important;
                box-shadow: 0 12px 26px rgba(14, 124, 115, 0.25);
            }

            div[data-testid="stButton"] button p,
            div[data-testid="stButton"] button span {
                color: inherit !important;
            }

            [data-testid="stAlert"] {
                border-radius: 8px;
            }

            [data-testid="stSlider"] [role="slider"] {
                background: var(--accent) !important;
            }

            @media (max-width: 720px) {
                .metric-strip {
                    grid-template-columns: 1fr;
                }

                .product-card {
                    grid-template-columns: 1fr;
                }

                .product-copy {
                    padding: 0 1rem 1rem;
                }

                .product-image {
                    border-right: 0;
                    border-bottom: 1px solid rgba(23, 32, 42, 0.08);
                    height: 210px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Gemini Product Recommender",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    dataset_path = "flipkart_com-ecommerce_sample.csv"
    df = load_data(dataset_path)

    if df is None:
        return

    refined_df = preprocess_data(df)
    product_count = len(refined_df)
    category_count = refined_df["primary_category"].nunique()
    median_price = refined_df["discounted_price"].median()

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <h2>ShopLens</h2>
            <p>Natural-language recommendations powered by Gemini intent and catalog ranking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    option = st.sidebar.radio("View", ("Product Recommendation", "Data Analysis"))
    st.sidebar.caption("Uses GEMINI_API_KEY for intent extraction and local NLP ranking over the catalog.")

    st.markdown(
        f"""
        <header class="app-shell">
            <h1 class="app-title">Gemini powered product recommendations.</h1>
            <p class="app-subtitle">
                Search naturally, compare quickly, and get products ranked by meaning, price, discount, and catalog fit.
            </p>
            <div class="metric-strip">
                <div class="metric-tile"><strong>{product_count:,}</strong><span>catalog products</span></div>
                <div class="metric-tile"><strong>{category_count:,}</strong><span>top-level categories</span></div>
                <div class="metric-tile"><strong>Rs. {median_price:,.0f}</strong><span>median sale price</span></div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    if option == "Data Analysis":
        display_data_analysis(refined_df)
    else:
        display_product_recommendation(refined_df)


if __name__ == "__main__":
    main()
