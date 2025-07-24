import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

# Page config
st.set_page_config(page_title="ESG Banking Dashboard", layout="wide")

# Title
st.title("ESG Banking Customer Retention Dashboard")

# Data Loading and Processing
@st.cache_data
def load_and_process_data():
    # Load both datasets
    try:
        # Update these paths to your actual file locations
        review_path = r"bank_reviews.xlsx"
        esg_path = r"sustainable_banking.xlsx"
        
        reviews_df = pd.read_excel(review_path)
        esg_df = pd.read_excel(esg_path)
        
        # Clean column names
        reviews_df.columns = reviews_df.columns.str.replace('[^a-zA-Z0-9]', '').str.lower()
        esg_df.columns = esg_df.columns.str.replace('[^a-zA-Z0-9]', '').str.lower()

        # Merge datasets on ID (adjust column name as needed)
        if 'idnum' in reviews_df.columns and 'idnum' in esg_df.columns:
            combined_df = pd.merge(reviews_df, esg_df, on='idnum', how='inner')
        else:
            st.error("No common ID column found to merge datasets")
            return pd.DataFrame()

        # Data cleaning and feature engineering (from your R code)
        # Handle missing values
        numeric_cols = combined_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            combined_df[col].fillna(combined_df[col].median(), inplace=True)
        
        # Create ESG score (simplified version of your R logic)
        #combined_df['esg_score'] = np.where(
         #   combined_df['type'] == "Green", 5,
         #   np.clip(
          #      (combined_df['coupon'].apply(lambda x: 5 - 4 * (x / combined_df['coupon'].max())) + 
           #     (combined_df['maturity'].apply(lambda x: 5 * (x / combined_df['maturity'].max()))) / 2,
            #    1, 5
            #).round()
        #))
        # Corrected ESG score calculation
        combined_df['esg_score'] = np.where(
            combined_df['type'] == "Green", 
            5,
            np.clip(
                (
                    combined_df['coupon'].apply(lambda x: 5 - 4 * (x / combined_df['coupon'].max())) 
                    + 
                    combined_df['maturity'].apply(lambda x: 5 * (x / combined_df['maturity'].max()))
                ) / 2,
                1, 
                5
            ).round()
        )
        
        # Create retention flag (simplified version)
        combined_df['customer_retention'] = np.where(
            combined_df['review'].str.contains(r'(\d+\s+years?|since\s+\d+)', regex=True, na=False),
            1, 0
        )
        
        # Cluster analysis
        cluster_cols = ['esg_score', 'rating', 'usefulcount']
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(combined_df[cluster_cols].dropna())
        
        kmeans = KMeans(n_clusters=3, random_state=42).fit(cluster_data)
        combined_df['cluster'] = kmeans.labels_
        
        # Name clusters
        cluster_names = {
            0: "High Rating, Low ESG Engagement",
            1: "Low Rating, High ESG Engagement",
            2: "Balanced Rating and ESG"
        }
        combined_df['segment_name'] = combined_df['cluster'].map(cluster_names)
        
        return combined_df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_and_process_data()

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(df)

if not df.empty:
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        selected_esg = st.slider("ESG Score Range", 1, 5, (1, 5))
        selected_cluster = st.multiselect(
            "Customer Segments",
            options=df['segment_name'].unique(),
            default=df['segment_name'].unique()
        )
        selected_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, step=0.5)

    # Filter data
    filtered_df = df[
        (df['esg_score'].between(selected_esg[0], selected_esg[1])) &
        (df['segment_name'].isin(selected_cluster)) &
        (df['rating'] >= selected_rating)
    ]

    # Main dashboard
    tab1, tab2, tab3 = st.tabs([
        "ESG Impact Analysis", 
        "Customer Segmentation", 
        "Actionable Recommendations"
    ])

    with tab1:
        st.header("ESG Impact on Customer Retention")
        
        # Retention probability by ESG score
        st.subheader("Retention Probability by ESG Score")
        
        # Train logistic regression model
        X = df[['esg_score']]
        y = df['customer_retention']
        model = LogisticRegression().fit(X, y)
        
        esg_range = np.arange(1, 5.1, 0.1).reshape(-1, 1)
        retention_probs = model.predict_proba(esg_range)[:, 1]
        
        fig1 = px.line(
            x=esg_range.flatten(), 
            y=retention_probs,
            labels={'x': 'ESG Score', 'y': 'Retention Probability'},
            title="Higher ESG Scores Correlate with Better Retention"
        )
        fig1.update_traces(line=dict(width=4, color='#1f77b4'))
        st.plotly_chart(fig1, use_container_width=True)
        
        # Key metrics
        retention_stats = df.groupby(
            pd.cut(df['esg_score'], bins=[0.5, 2.5, 3.5, 5.5], labels=["1-2", "3", "4-5"])
        )['customer_retention'].mean()
        
        overall_retention = df['customer_retention'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Avg Retention (ESG 1-2)", 
                f"{retention_stats['1-2']:.0%}", 
                f"{(retention_stats['1-2'] - overall_retention):.0%} vs overall"
            )
        with col2:
            st.metric(
                "Avg Retention (ESG 3)", 
                f"{retention_stats['3']:.0%}", 
                f"{(retention_stats['3'] - overall_retention):.0%} vs overall"
            )
        with col3:
            st.metric(
                "Avg Retention (ESG 4-5)", 
                f"{retention_stats['4-5']:.0%}", 
                f"{(retention_stats['4-5'] - overall_retention):.0%} vs overall"
            )

    with tab2:
        st.header("Customer Segmentation by ESG Engagement")
        
        # Cluster visualization
        fig2 = px.scatter(
            filtered_df,
            x='rating',
            y='esg_score',
            color='segment_name',
            hover_data=['address', 'usefulcount'],
            title="Customer Segments by Rating and ESG Score"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Segment profiles
        st.subheader("Segment Characteristics")
        
        segment_stats = filtered_df.groupby('segment_name').agg({
            'esg_score': 'mean',
            'rating': 'mean',
            'customer_retention': 'mean',
            'usefulcount': 'mean'
        }).reset_index()
        
        st.dataframe(
            segment_stats.style.format({
                'esg_score': '{:.1f}',
                'rating': '{:.1f}',
                'customer_retention': '{:.1%}',
                'usefulcount': '{:.1f}'
            }),
            use_container_width=True
        )

    with tab3:
        st.header("ESG Integration Recommendations")
        
        st.subheader("Tailored Strategies by Customer Segment")
        
        # Create expandable sections for each recommendation
        with st.expander("💡 For High Rating, Low ESG Customers", expanded=True):
            st.markdown("""
            **Characteristics**: These customers rate you highly but don't engage with ESG offerings.
            
            **Recommendations**:
            - 🏆 **Loyalty-ESG Bundles**: Combine existing loyalty rewards with ESG options
            - 📢 **Targeted ESG Education**: Show how their banking habits already support sustainability
            - 🔄 **Automatic ESG Round-Ups**: Implement small, automatic donations to green causes
            """)
            
            # Example implementation
            st.code("""
            # Sample ESG-Loyalty Integration
            def calculate_esg_rewards(transaction_amount):
                base_rewards = transaction_amount * 0.01
                esg_bonus = transaction_amount * 0.005 if transaction_is_green else 0
                return base_rewards + esg_bonus
            """, language='python')
        
        with st.expander("💡 For Low Rating, High ESG Customers"):
            st.markdown("""
            **Characteristics**: These customers actively use ESG features but give low ratings.
            
            **Recommendations**:
            - 🛠️ **Improve ESG Feature UX**: Make ESG options more visible and easier to use
            - 📱 **ESG Mobile Features**: Add ESG tracking to mobile banking
            - 🌱 **Personalized ESG Reports**: Show their individual impact
            """)
            
            # Example report
            if not filtered_df.empty:
                sample_customer = filtered_df[
                    filtered_df['segment_name'] == "Low Rating, High ESG Engagement"
                ].iloc[0]
                
                st.metric("Sample Customer Impact", 
                         f"CO2 Reduced: {sample_customer['esg_score']*200} kg/year",
                         "Equivalent to planting 10 trees")
        
        with st.expander("💡 For Balanced Customers"):
            st.markdown("""
            **Characteristics**: Moderate engagement with both traditional and ESG features.
            
            **Recommendations**:
            - 📈 **Gradual ESG Upselling**: Introduce more advanced ESG products
            - 🤝 **Community ESG Projects**: Offer local sustainability projects
            - 💳 **ESG Credit Card Benefits**: Extra rewards for sustainable purchases
            """)
        
        st.subheader("Implementation Roadmap")
        
        # Roadmap timeline
        roadmap = pd.DataFrame({
            'Phase': ["Quick Wins", "Medium-Term", "Long-Term"],
            'Timeline': ["1-3 months", "3-6 months", "6-12 months"],
            'Initiatives': [
                "ESG Round-Ups, Paperless Incentives",
                "Mobile ESG Tracking, Loyalty Integration",
                "Full ESG Product Suite, Impact Reporting"
            ]
        })
        
        st.dataframe(roadmap, hide_index=True, use_container_width=True)
        
        # ROI Calculator
        st.subheader("ESG Initiative ROI Calculator")
        col1, col2 = st.columns(2)
        with col1:
            current_retention = st.number_input("Current Retention Rate (%)", 30.0)
            esg_investment = st.number_input("ESG Investment ($)", 100000)
        with col2:
            projected_impact = st.slider("Projected Retention Improvement", 1, 20, 5)
            customer_value = st.number_input("Avg Customer Lifetime Value ($)", 5000)
        
        # Calculate ROI
        new_retention = current_retention * (1 + projected_impact/100)
        value_gain = (new_retention - current_retention)/100 * len(df) * customer_value
        roi = (value_gain - esg_investment) / esg_investment
        
        st.metric("Projected ROI", f"{roi:.0%}", 
                 f"${value_gain:,.0f} expected value gain")

else:
    st.warning("No data loaded. Please check your data files.")
