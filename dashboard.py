import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
from PIL import Image
import io
import base64
from datetime import datetime
import time

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')

# Set page config with modern theme
st.set_page_config(
    page_title="📱 Play Store Insights Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #07111D;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0D1B2A, #4a4e69) !important;
        color: white !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] .st-c7 {
        color: white !important;
    }
    
    /* Cards */
    .card {
        background: black;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: white;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: black;
        color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #3a3a3a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: black;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #6B7AA1;
        color: black;
        transition: 3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0D1B2A;
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        color: white !important;
    }
    
    /* Select boxes */
    [data-baseweb="select"] {
    background: pink;
        border-radius: 8px !important;
    }
    
    /* Custom card for visualizations */
    .viz-card {
        background: black;
        border-radius: 10px;
        color: white;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load the datasets with progress indicator
@st.cache_data
def load_data():
    with st.spinner('Loading data... This may take a moment'):
        try:
            playstore_df = pd.read_csv('googleplaystore.csv')
            reviews_df = pd.read_csv('apps_reviews_40.csv')
            
            # Simulate loading time for demo
            time.sleep(2)
            
            st.success("Data loaded successfully!")
            return playstore_df, reviews_df
            
        except FileNotFoundError as e:
            st.warning(f"Error loading files: {e}")
            # For demonstration, create minimal dataframes
            playstore_data = """App,Category,Rating,Reviews,Size,Installs,Type,Price,Content Rating,Genres,Last Updated,Current Ver,Android Ver
Photo Editor & Candy Camera & Grid & ScrapBook,ART_AND_DESIGN,4.1,159,19M,10,000+,Free,0,Everyone,Art & Design,January 7, 2018,1.0.0,4.0.3 and up
Coloring book moana,ART_AND_DESIGN,3.9,967,14M,500,000+,Free,0,Everyone,Art & Design;Pretend Play,January 15, 2018,2.0.0,4.0.3 and up"""
            playstore_df = pd.read_csv(pd.compat.StringIO(playstore_data))
            
            reviews_data = """reviewId,userName,userImage,content,score,thumbsUpCount,reviewCreatedVersion,at,replyContent,repliedAt,appVersion,package
00351e82-18ff-445f-ac54-ad7ba297aee6,Musa Laberia,https://play-lh.googleusercontent.com/a/ACg8ocK3HLxf88LWfnbs-yicI2E-JvVARQk2f96yKNOm1QsOuytLNw=mo,Not responding,5,0,2.25.17.80,6/25/2025 15:42,,,2.25.17.80,com.whatsapp
77b577b5-8845-467a-a136-2e1ad6ba428b,Alok Rout,https://play-lh.googleusercontent.com/a-/ALV-UjWjfmXJP1WhQCeMvdgvby29vRkNpV_trifdwLDrl27l9faXXEx1,outgoing msg not working,1,0,2.25.1.75,6/25/2025 15:42,,,2.25.1.75,com.whatsapp
87d0af91-bd00-47b4-b959-42c4d45b7b43,Jeevan Nair 0151,https://play-lh.googleusercontent.com/a-/ALV-UjURJamyLHIMS02jI6IPax7Dmat10NKc0ksl5-CZW-zmTqbj--U,App was great and easy to use,5,0,2.23.14.79,6/25/2025 15:41,,,2.23.14.79,com.whatsapp"""
            reviews_df = pd.read_csv(pd.compat.StringIO(reviews_data))
    
    return playstore_df, reviews_df

playstore_df, reviews_df = load_data()

# Data Cleaning and Preprocessing with progress bar
@st.cache_data
def clean_playstore_data(df):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    status_text.text("Cleaning data... Step 1/7: Handling missing values")
    progress_bar.progress(15)
    # Handle missing values - only drop rows with missing Rating
    df = df.dropna(subset=['Rating'])
    
    status_text.text("Cleaning data... Step 2/7: Cleaning Installs column")
    progress_bar.progress(30)
    # Clean 'Installs' column
    df['Installs'] = (
        df['Installs']
        .astype(str)
        .str.replace('+', '')
        .str.replace(',', '')
        .replace('Free', '0')
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .astype(int)
    )
    
    status_text.text("Cleaning data... Step 3/7: Cleaning Size column")
    progress_bar.progress(45)
    # Clean 'Size' column
    df['Size'] = (
        df['Size']
        .astype(str)
        .str.replace('M', 'e6')
        .str.replace('k', 'e3')
        .str.replace('Varies with device', 'NaN')
        .apply(pd.to_numeric, errors='coerce')
    )
    
    status_text.text("Cleaning data... Step 4/7: Cleaning Price column")
    progress_bar.progress(60)
    # Clean 'Price' column
    df['Price'] = (
        df['Price']
        .astype(str)
        .str.replace('$', '')
        .replace('Everyone', '0')
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )
    
    status_text.text("Cleaning data... Step 5/7: Converting Reviews")
    progress_bar.progress(75)
    # Convert 'Reviews' to numeric
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0)
    
    status_text.text("Cleaning data... Step 6/7: Converting dates")
    progress_bar.progress(90)
    # Convert 'Last Updated' to datetime
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    
    status_text.text("Cleaning data... Step 7/7: Final transformations")
    # Create a binary column for Free/Paid
    df['Is_Free'] = df['Type'].apply(lambda x: 1 if x == 'Free' else 0)
    
    progress_bar.progress(100)
    status_text.text("Data cleaning complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return df

@st.cache_data
def clean_reviews_data(df):
    # Convert date column to datetime
    df['at'] = pd.to_datetime(df['at'])
    
    # Handle missing values
    df['content'] = df['content'].fillna('')
    
    return df

playstore_df = clean_playstore_data(playstore_df)
reviews_df = clean_reviews_data(reviews_df)

# Data Transformation
@st.cache_data
def transform_data(df):
    # Create log-transformed installs
    df['Log_Installs'] = np.log1p(df['Installs'])
    
    # Categorize ratings
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']
    df['Rating_Category'] = pd.cut(df['Rating'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate revenue estimate (for paid apps)
    df['Revenue_Estimate'] = df['Price'] * df['Installs']
    
    # Extract year from last updated
    df['Last_Updated_Year'] = df['Last Updated'].dt.year
    
    return df

playstore_df = transform_data(playstore_df)

# Sentiment Analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

@st.cache_data
def perform_sentiment_analysis(reviews_df):
    with st.spinner('Performing sentiment analysis on reviews...'):
        reviews_df['Sentiment'] = reviews_df['content'].apply(analyze_sentiment)
        
        # Categorize sentiment
        reviews_df['Sentiment_Label'] = pd.cut(reviews_df['Sentiment'], 
                                             bins=[-1, -0.5, 0, 0.5, 1],
                                             labels=['Negative', 'Somewhat Negative', 
                                                    'Somewhat Positive', 'Positive'])
        
        # Add emoji based on sentiment
        def sentiment_to_emoji(sentiment):
            if sentiment > 0.5:
                return "😊"
            elif sentiment > 0:
                return "🙂"
            elif sentiment > -0.5:
                return "😐"
            else:
                return "😞"
        
        reviews_df['Sentiment_Emoji'] = reviews_df['Sentiment'].apply(sentiment_to_emoji)
        
        return reviews_df

if not reviews_df.empty:
    reviews_df = perform_sentiment_analysis(reviews_df)
    
    # Merge sentiment with playstore data
    avg_sentiment = reviews_df.groupby('package')['Sentiment'].mean().reset_index()
    playstore_df = playstore_df.merge(avg_sentiment, left_on='App', right_on='package', how='left')
    playstore_df['Sentiment'] = playstore_df['Sentiment'].fillna(0)

# New Feature: App Age Calculation
@st.cache_data
def calculate_app_age(df):
    df['Last_Updated_Year'] = df['Last Updated'].dt.year
    current_year = datetime.now().year
    df['App_Age'] = current_year - df['Last_Updated_Year']
    return df

playstore_df = calculate_app_age(playstore_df)

# New Feature: Price Category
@st.cache_data
def add_price_category(df):
    df['Price_Category'] = pd.cut(df['Price'],
                                bins=[-1, 0, 1, 5, 10, 20, 50, float('inf')],
                                labels=['Free', '$0-$1', '$1-$5', '$5-$10', 
                                       '$10-$20', '$20-$50', '$50+'])
    return df

playstore_df = add_price_category(playstore_df)

# Modern App Card Component
def app_card(app_data):
    card_html = f"""
    <div style="
        background: black;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: white;">{app_data['App']}</h4>
                <p style="margin: 5px 0; color: white; font-size: 0.9em;">{app_data['Category']}</p>
            </div>
            <div style="
                background: linear-gradient(135deg, #6e8efb, #a777e3);
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-weight: bold;
            ">
                {app_data['Rating']:.1f} ⭐
            </div>
        </div>
        <div style="margin-top: 10px; display: flex; justify-content: space-between;">
            <span style="font-size: 0.8em; color: grey;">📥 {app_data['Installs']:,}+</span>
            <span style="font-size: 0.8em; color: grey;">💰 {'Free' if app_data['Is_Free'] else '$'+str(app_data['Price'])}</span>
            <span style="font-size: 0.8em; color: grey;">🔄 {app_data['Last_Updated_Year']}</span>
        </div>
    </div>
    """
    return card_html

# Streamlit App
def main():
    st.title("📱 Play Store Insights Pro")
    st.markdown("""
    <style>
        .title {
            color: #6e8efb;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
        <h2 style='color: white; text-align: center;'>🔍 Filters</h2>
        """, unsafe_allow_html=True)
        
        selected_category = st.selectbox(
            "Select Category", 
            ['All'] + sorted(playstore_df['Category'].unique().tolist()),
            key='category_filter'
        )
        
        min_rating, max_rating = st.slider(
            "Rating Range",
            0.0, 5.0, (0.0, 5.0),
            key='rating_filter'
        )
        
        install_range = st.slider(
            "Installs Range (millions)",
            0, int(playstore_df['Installs'].max() / 1000000),
            (0, int(playstore_df['Installs'].max() / 1000000)),
            key='installs_filter'
        )
        
        price_filter = st.multiselect(
            "Price Category",
            options=playstore_df['Price_Category'].unique(),
            default=playstore_df['Price_Category'].unique(),
            key='price_filter'
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: white;">
            <p><>Explore app data and reviews</p>
            <p><> Analyze trends and patterns</p>
            <p>🤖 Predict app success</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filter data based on selections
    filtered_df = playstore_df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    filtered_df = filtered_df[
        (filtered_df['Rating'] >= min_rating) & 
        (filtered_df['Rating'] <= max_rating) &
        (filtered_df['Installs'] >= install_range[0] * 1000000) &
        (filtered_df['Installs'] <= install_range[1] * 1000000) &
        (filtered_df['Price_Category'].isin(price_filter))
    ]
    
    # Overview Metrics with modern cards
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Apps", f"{len(filtered_df):,}", help="Number of apps in current selection")
    with col2:
        avg_rating = filtered_df['Rating'].mean()
        rating_delta = avg_rating - playstore_df['Rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}", 
                 f"{rating_delta:.2f} vs overall", 
                 delta_color="normal",
                 help="Average rating compared to all apps")
    with col3:
        total_installs = filtered_df['Installs'].sum()
        st.metric("Total Installs", f"{total_installs/1000000:.1f}M", 
                 help="Sum of all installs in current selection")
    with col4:
        paid_count = len(filtered_df[filtered_df['Type'] == 'Paid'])
        paid_pct = (paid_count / len(filtered_df)) * 100
        st.metric("Paid Apps", f"{paid_count:,}", 
                 f"{paid_pct:.1f}% of total", 
                 help="Number and percentage of paid apps")
    
    # Tabs for different sections with modern styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📱 App Explorer", 
        "📈 Market Trends", 
        "💬 Reviews Insights", 
        "💰 Revenue Analysis",
        "🤖 AI Predictor"
    ])
    
    with tab1:
        st.subheader("App Explorer")
        
        # Top Apps section
        st.markdown("### 🏆 Top Performing Apps")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top_by_rating = filtered_df.sort_values('Rating', ascending=False).head(5)
            st.markdown("##### Highest Rated")
            for _, app in top_by_rating.iterrows():
                st.markdown(app_card(app), unsafe_allow_html=True)
        
        with col2:
            top_by_installs = filtered_df.sort_values('Installs', ascending=False).head(5)
            st.markdown("##### Most Installed")
            for _, app in top_by_installs.iterrows():
                st.markdown(app_card(app), unsafe_allow_html=True)
        
        with col3:
            if 'Revenue_Estimate' in filtered_df.columns:
                top_by_revenue = filtered_df[filtered_df['Type'] == 'Paid'].sort_values('Revenue_Estimate', ascending=False).head(5)
                st.markdown("##### Highest Revenue (Paid)")
                for _, app in top_by_revenue.iterrows():
                    st.markdown(app_card(app), unsafe_allow_html=True)
        
        # App Distribution
        st.markdown("### 📊 App Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Rating Distribution")
            fig = px.histogram(
                filtered_df, 
                x='Rating', 
                nbins=20,
                color_discrete_sequence=['#6e8efb'],
                marginal='box'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Free vs Paid Apps")
            fig = px.pie(
                filtered_df,
                names='Type',
                color_discrete_sequence=['#6e8efb', '#a777e3']
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hole=.4
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Market Trends Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Top Categories by App Count")
            top_categories = filtered_df['Category'].value_counts().head(10)
            fig = px.bar(
                top_categories,
                orientation='h',
                color=top_categories.values,
                color_continuous_scale='blues',
                labels={'value': 'Number of Apps', 'index': 'Category'}
            )
            fig.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Top Categories by Installs")
            top_categories_installs = filtered_df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                top_categories_installs,
                orientation='h',
                color=top_categories_installs.values,
                color_continuous_scale='purples',
                labels={'value': 'Total Installs (millions)', 'index': 'Category'}
            )
            fig.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("##### App Releases Over Time")
        yearly_releases = filtered_df.groupby('Last_Updated_Year').size().reset_index(name='Count')
        fig = px.line(
            yearly_releases,
            x='Last_Updated_Year',
            y='Count',
            markers=True,
            color_discrete_sequence=['#6e8efb']
        )
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Apps',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("##### Rating vs Installs")
        fig = px.scatter(
            filtered_df, 
            x='Rating', 
            y='Installs',
            size='Reviews',
            color='Category',
            hover_name='App',
            log_y=True,
            size_max=30,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if not reviews_df.empty:
            st.subheader("Reviews Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Sentiment Distribution")
                fig = px.pie(
                    reviews_df,
                    names='Sentiment_Label',
                    color='Sentiment_Label',
                    color_discrete_map={
                        'Negative': '#ff4d4d',
                        'Somewhat Negative': '#ff9999',
                        'Somewhat Positive': '#99cc99',
                        'Positive': '#66b266'
                    },
                    hole=0.4
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### Sentiment Over Time")
                reviews_df['date'] = reviews_df['at'].dt.date
                daily_sentiment = reviews_df.groupby('date')['Sentiment'].mean().reset_index()
                fig = px.line(
                    daily_sentiment,
                    x='date',
                    y='Sentiment',
                    color_discrete_sequence=['#6e8efb']
                )
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Average Sentiment',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### Review Word Cloud")
            text = " ".join(review for review in reviews_df['content'])
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='Blues'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("##### Recent Reviews")
            st.dataframe(
                reviews_df[['userName', 'content', 'score', 'Sentiment_Emoji']].head(10),
                column_config={
                    "userName": "User",
                    "content": "Review",
                    "score": "Rating",
                    "Sentiment_Emoji": "Sentiment"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No reviews data available for analysis")
    
    with tab4:
        st.subheader("Revenue Analysis")
        
        if 'Revenue_Estimate' in filtered_df.columns:
            paid_apps = filtered_df[filtered_df['Type'] == 'Paid']
            
            if not paid_apps.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Price Distribution")
                    fig = px.histogram(
                        paid_apps,
                        x='Price',
                        nbins=20,
                        color_discrete_sequence=['#a777e3']
                    )
                    fig.update_layout(
                        xaxis_title='Price ($)',
                        yaxis_title='Number of Apps',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### Top Revenue Generating Apps")
                    top_revenue = paid_apps.sort_values('Revenue_Estimate', ascending=False).head(10)
                    fig = px.bar(
                        top_revenue,
                        x='App',
                        y='Revenue_Estimate',
                        color='Revenue_Estimate',
                        color_continuous_scale='purples',
                        labels={'Revenue_Estimate': 'Revenue ($)'}
                    )
                    fig.update_layout(
                        xaxis_title='App',
                        yaxis_title='Estimated Revenue ($)',
                        coloraxis_showscale=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Price vs Installs")
                fig = px.scatter(
                    paid_apps,
                    x='Price',
                    y='Installs',
                    size='Revenue_Estimate',
                    color='Category',
                    hover_name='App',
                    log_y=True,
                    size_max=30,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(
                    xaxis_title='Price ($)',
                    yaxis_title='Installs',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No paid apps in current selection")
        else:
            st.warning("Revenue data not available")
    
    with tab5:
        st.subheader("App Success Predictor")
        
        if len(playstore_df) > 100:
            # Prepare ML data
            @st.cache_data
            def prepare_ml_data(df):
                features = ['Rating', 'Reviews', 'Size', 'Is_Free', 'Price', 'Sentiment', 'App_Age']
                target = 'Log_Installs'
                
                le = LabelEncoder()
                df['Category_Encoded'] = le.fit_transform(df['Category'])
                features.append('Category_Encoded')
                
                ml_df = df.dropna(subset=features + [target])
                X = ml_df[features]
                y = ml_df[target]
                
                return X, y, le
            
            X, y, le = prepare_ml_data(playstore_df)
            
            # Train model with progress indicator
            with st.spinner('Training machine learning model...'):
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            st.markdown("##### Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with col2:
                st.metric("R-squared Score", f"{r2:.4f}")
            
            st.markdown("##### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='blues'
            )
            fig.update_layout(
                yaxis_title='Feature',
                xaxis_title='Importance',
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction interface
            st.markdown("##### Make a Prediction")
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rating = st.slider("Rating", 0.0, 5.0, 4.0, step=0.1)
                    reviews = st.number_input("Number of Reviews", min_value=0, value=1000, step=100)
                    size = st.number_input("Size (in MB)", min_value=0, value=10, step=1)
                
                with col2:
                    is_free = st.selectbox("Is Free?", [True, False])
                    price = st.number_input("Price ($)", min_value=0.0, value=0.0, step=0.99, disabled=is_free)
                    sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.5, step=0.1)
                
                with col3:
                    category = st.selectbox("Category", playstore_df['Category'].unique())
                    app_age = st.number_input("App Age (years)", min_value=0, value=2, step=1)
                
                submitted = st.form_submit_button("Predict Installs")
                if submitted:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'Rating': [rating],
                        'Reviews': [reviews],
                        'Size': [size * 1e6],  # Convert MB to bytes
                        'Is_Free': [1 if is_free else 0],
                        'Price': [0 if is_free else price],
                        'Sentiment': [sentiment],
                        'App_Age': [app_age],
                        'Category_Encoded': [le.transform([category])[0]]
                    })
                    
                    # Make prediction
                    prediction = model.predict(input_data)
                    predicted_installs = np.expm1(prediction[0])
                    
                    # Show result with animation
                    with st.spinner('Calculating prediction...'):
                        time.sleep(1)
                        
                    st.success(f"""
                    ### Predicted Installs: {predicted_installs:,.0f}
                    
                    This app is predicted to have approximately **{predicted_installs:,.0f}** installs based on the provided parameters.
                    """)
        else:
            st.warning("Insufficient data for meaningful machine learning modeling")

if __name__ == "__main__":
    main()
