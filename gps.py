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
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import base64

nltk.download('vader_lexicon')
nltk.download('punkt')

try:
    playstore_df = pd.read_csv('googleplaystore.csv')
    reviews_df = pd.read_csv('apps_reviews_40.csv')  
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    playstore_data = """App,Category,Rating,Reviews,Size,Installs,Type,Price,Content Rating,Genres,Last Updated,Current Ver,Android Ver
Photo Editor & Candy Camera & Grid & ScrapBook,ART_AND_DESIGN,4.1,159,19M,10,000+,Free,0,Everyone,Art & Design,January 7, 2018,1.0.0,4.0.3 and up
Coloring book moana,ART_AND_DESIGN,3.9,967,14M,500,000+,Free,0,Everyone,Art & Design;Pretend Play,January 15, 2018,2.0.0,4.0.3 and up"""
    playstore_df = pd.read_csv(pd.compat.StringIO(playstore_data))
    
    reviews_data = """reviewId,userName,userImage,content,score,thumbsUpCount,reviewCreatedVersion,at,replyContent,repliedAt,appVersion,package
00351e82-18ff-445f-ac54-ad7ba297aee6,Musa Laberia,https://play-lh.googleusercontent.com/a/ACg8ocK3HLxf88LWfnbs-yicI2E-JvVARQk2f96yKNOm1QsOuytLNw=mo,Not responding,5,0,2.25.17.80,6/25/2025 15:42,,,2.25.17.80,com.whatsapp
77b577b5-8845-467a-a136-2e1ad6ba428b,Alok Rout,https://play-lh.googleusercontent.com/a-/ALV-UjWjfmXJP1WhQCeMvdgvby29vRkNpV_trifdwLDrl27l9faXXEx1,outgoing msg not working,1,0,2.25.1.75,6/25/2025 15:42,,,2.25.1.75,com.whatsapp
87d0af91-bd00-47b4-b959-42c4d45b7b43,Jeevan Nair 0151,https://play-lh.googleusercontent.com/a-/ALV-UjURJamyLHIMS02jI6IPax7Dmat10NKc0ksl5-CZW-zmTqbj--U,App was great and easy to use,5,0,2.23.14.79,6/25/2025 15:41,,,2.23.14.79,com.whatsapp"""
    reviews_df = pd.read_csv(pd.compat.StringIO(reviews_data))

def clean_playstore_data(df):
 
    df = df.copy()
    

    df = df.dropna(subset=['Rating'])
    

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
    
    df['Size'] = (
        df['Size']
        .astype(str)
        .str.replace('M', 'e6') 
        .str.replace('k', 'e3')  
        .str.replace('Varies with device', 'NaN')  
        .apply(pd.to_numeric, errors='coerce') 
    )
    
    df['Price'] = (
        df['Price']
        .astype(str)
        .str.replace('$', '')
        .replace('Everyone', '0')  
        .apply(pd.to_numeric, errors='coerce')  
        .fillna(0)  
    )
    
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0)
    
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    
    df['Is_Free'] = df['Type'].apply(lambda x: 1 if x == 'Free' else 0)
    
    return df

def clean_reviews_data(df):
    df['at'] = pd.to_datetime(df['at'])
    
    df['content'] = df['content'].fillna('')
    
    return df

playstore_df = clean_playstore_data(playstore_df)
reviews_df = clean_reviews_data(reviews_df)

def transform_data(df):
    df['Log_Installs'] = np.log1p(df['Installs'])
    
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']
    df['Rating_Category'] = pd.cut(df['Rating'], bins=bins, labels=labels, include_lowest=True)
    
    df['Revenue_Estimate'] = df['Price'] * df['Installs']
    
    return df

playstore_df = transform_data(playstore_df)

# Exploratory Data Analysis
def perform_eda(df):
    print("Summary Statistics:")
    print(df.describe())
    
    plt.figure(figsize=(12, 6))
    top_categories = df['Category'].value_counts().head(10)
    sns.barplot(x=top_categories.values, y=top_categories.index)
    plt.title('Top 10 App Categories')
    plt.xlabel('Number of Apps')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rating'], bins=20, kde=True)
    plt.title('Distribution of App Ratings')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Rating', y='Log_Installs', data=df)
    plt.title('Rating vs Log Transformed Installs')
    plt.show()

perform_eda(playstore_df)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def perform_sentiment_analysis(reviews_df):
    reviews_df['Sentiment'] = reviews_df['content'].apply(analyze_sentiment)
    
    reviews_df['Sentiment_Label'] = pd.cut(reviews_df['Sentiment'], 
                                         bins=[-1, -0.5, 0, 0.5, 1],
                                         labels=['Negative', 'Somewhat Negative', 
                                                'Somewhat Positive', 'Positive'])
    
    plt.figure(figsize=(10, 6))
    reviews_df['Sentiment_Label'].value_counts().plot(kind='bar')
    plt.title('Distribution of Review Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
    
    return reviews_df

if not reviews_df.empty:
    reviews_df = perform_sentiment_analysis(reviews_df)
    
    avg_sentiment = reviews_df.groupby('package')['Sentiment'].mean().reset_index()
    playstore_df = playstore_df.merge(avg_sentiment, left_on='App', right_on='package', how='left')
    playstore_df['Sentiment'] = playstore_df['Sentiment'].fillna(0)

def create_interactive_visualizations(df):
    fig1 = px.bar(df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10),
                 title='Top Categories by Total Installs',
                 labels={'value': 'Total Installs', 'index': 'Category'})
    
    fig2 = px.scatter(df, x='Rating', y='Log_Installs', color='Type',
                     title='Rating vs Log Transformed Installs',
                     hover_data=['App', 'Category'])
    
    fig3 = px.histogram(df[df['Type'] == 'Paid'], x='Price',
                       title='Price Distribution for Paid Apps',
                       nbins=20)
    
    fig1.show()
    fig2.show()
    fig3.show()

create_interactive_visualizations(playstore_df)

def prepare_ml_data(df):
    features = ['Rating', 'Reviews', 'Size', 'Is_Free', 'Price', 'Sentiment']
    target = 'Log_Installs'
    
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Category'])
    
    features.append('Category_Encoded')
    
    ml_df = df.dropna(subset=features + [target])
    
    X = ml_df[features]
    y = ml_df[target]
    
    return X, y

def train_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for App Success Prediction')
    plt.show()
    
    return model

if len(playstore_df) > 100:  
    X, y = prepare_ml_data(playstore_df)
    model = train_ml_model(X, y)
else:
    print("Insufficient data for meaningful machine learning modeling")

class PlaystoreDashboard:
    def __init__(self, root, playstore_df, reviews_df):
        self.root = root
        self.playstore_df = playstore_df
        self.reviews_df = reviews_df
        self.root.title("Google Play Store Analytics Dashboard")
        
        self.tab_control = ttk.Notebook(root)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text='Overview')
        self.tab_control.add(self.tab2, text='Category Analysis')
        self.tab_control.add(self.tab3, text='Reviews Analysis')
        
        self.tab_control.pack(expand=1, fill='both')
        
        self.create_overview_tab()
        self.create_category_tab()
        self.create_reviews_tab()
    
    def create_overview_tab(self):
        total_apps = len(self.playstore_df)
        avg_rating = self.playstore_df['Rating'].mean()
        total_installs = self.playstore_df['Installs'].sum()
        
        ttk.Label(self.tab1, text=f"Total Apps: {total_apps}").pack()
        ttk.Label(self.tab1, text=f"Average Rating: {avg_rating:.2f}").pack()
        ttk.Label(self.tab1, text=f"Total Installs: {total_installs:,}").pack()
        
        self.plot_to_tkinter(self.tab1, self.create_rating_dist_plot())
    
    def create_category_tab(self):
        top_categories = self.playstore_df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10)
        self.plot_to_tkinter(self.tab2, self.create_bar_plot(top_categories, 
                                                           'Top Categories by Installs',
                                                           'Category', 'Installs'))
        
        if 'Price' in self.playstore_df.columns:
            paid_apps = self.playstore_df[self.playstore_df['Type'] == 'Paid']
            if not paid_apps.empty:
                self.plot_to_tkinter(self.tab2, self.create_box_plot(paid_apps, 'Category', 'Price',
                                                                   'Price Distribution by Category'))
    
    def create_reviews_tab(self):
        if not self.reviews_df.empty and 'Sentiment' in self.reviews_df.columns:
            self.plot_to_tkinter(self.tab3, self.create_sentiment_plot())
            
            if 'score' in self.reviews_df.columns:
                self.plot_to_tkinter(self.tab3, self.create_sentiment_vs_rating_plot())
    
    def create_rating_dist_plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.playstore_df['Rating'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribution of App Ratings')
        return fig
    
    def create_bar_plot(self, data, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=data.values, y=data.index, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig
    
    def create_box_plot(self, data, x, y, title):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=data, x=x, y=y, ax=ax)
        ax.set_title(title)
        plt.xticks(rotation=45)
        return fig
    
    def create_sentiment_plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        self.reviews_df['Sentiment_Label'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Review Sentiments')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        return fig
    
    def create_sentiment_vs_rating_plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.reviews_df, x='score', y='Sentiment', ax=ax)
        ax.set_title('Sentiment vs User Rating')
        ax.set_xlabel('User Rating (1-5)')
        ax.set_ylabel('Sentiment Score')
        return fig
    
    def plot_to_tkinter(self, parent, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        
        label = ttk.Label(parent, image=photo)
        label.image = photo  
        label.pack()
        
        plt.close(fig)

root = tk.Tk()
app = PlaystoreDashboard(root, playstore_df, reviews_df)
root.mainloop()
