import streamlit as st
from utils.coordinator import generate_recommendation
from agents.rag_agent import rag_pipeline
from agents.forecasting_agent import get_symbol, forecast_pipeline


st.set_page_config(page_title="FinBot360 - Financial Advisor", layout="wide")

st.title("🚀 FinBot360: Multi-Agent Financial Advisor")

company = st.text_input("Enter the company name", "")

if st.button("Analyze"):
    if company.strip() == "":
        st.warning("Please enter a company name.")
    else:
        with st.spinner("Fetching data and analyzing, please wait... 🚀"):
            query = f"{company} in news"
            rag_pipeline(company)
            symbol = get_symbol(company)

            if not symbol:
                st.error("Could not find a valid stock symbol for this company. Please try a different name.")
            else:
                forecast = forecast_pipeline(symbol)
                result = generate_recommendation(symbol, forecast, query)
                st.success("Analysis Complete! 📊")
                st.subheader("📄 Final Recommendation")
                for k, v in result.items():
                    st.write(f"**{k.capitalize()}**: {v}")

