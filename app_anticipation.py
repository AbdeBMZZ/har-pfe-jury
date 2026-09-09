"""Standalone anticipation app; no legacy recognition checkpoint required."""
import streamlit as st
from src.ui.anticipation_panel import render
st.set_page_config(page_title='Anticipation HAR',layout='centered')
render()
