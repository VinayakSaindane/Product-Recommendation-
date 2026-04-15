# Gemini E-commerce Product Recommendation System

An NLP-first product recommender built with Streamlit, Gemini, and local catalog ranking. Users can type natural shopping requests such as:

```text
find me budget friendly shoes that are formal
black office footwear for men under 1000
comfortable cotton shorts for women at a low price
```

The app extracts shopping intent with Gemini when `GEMINI_API_KEY` is available, then ranks real products from the Flipkart catalog using local TF-IDF search, price awareness, discount strength, category matches, gender hints, brand hints, and style terms. If Gemini is not configured, the app still works through a local rule-based intent parser.

## Features

- Natural-language product search instead of rigid form inputs.
- Gemini-powered intent extraction using `GEMINI_API_KEY`.
- Local recommendation ranking grounded in the CSV dataset.
- Budget-aware, discount-aware, and category-aware scoring.
- Attractive Streamlit UI with product cards, images, prices, discounts, and match chips.
- Data analysis view for category price distribution and discount distribution.

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your-gemini-api-key
```

Optional:

```bash
GEMINI_MODEL=gemini-2.5-flash
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Dataset

The default dataset is `flipkart_com-ecommerce_sample.csv`. You can replace it with another CSV as long as it contains compatible product fields such as product name, category tree, price, image, description, brand, and product URL.

## Project Structure

```text
.
|-- app.py
|-- data_processing.py
|-- recommendation.py
|-- flipkart_com-ecommerce_sample.csv
|-- requirements.txt
`-- README.md
```
