from google_play_scraper import app, reviews
import pandas as pd
import time

app_packages = [
    # Social Media
    'com.whatsapp', 'com.instagram.android', 'com.facebook.katana', 'com.snapchat.android', 'com.twitter.android',
    # Entertainment
    'com.netflix.mediaclient', 'com.google.android.youtube', 'com.spotify.music', 'com.mxtech.videoplayer.ad',
    # Shopping
    'com.flipkart.android', 'in.amazon.mShop.android.shopping', 'com.myntra.android', 'com.snapdeal.main', 'com.ajio.android',
    # Finance
    'com.phonepe.app', 'com.google.android.apps.nbu.paisa.user', 'com.mobikwik_new', 'com.paytm', 'com.indianbank.bc',
    # Education
    'com.byjus.thelearningapp', 'com.duolingo', 'co.learncbse.app', 'org.khanacademy.android', 'com.toppr.student',
    # Health & Fitness
    'com.curefit.curefit', 'cc.fitness.fitnesscoach', 'fit.betterme', 'com.healthifyme.basic', 'com.google.android.apps.fitness',
    # Games
    'com.tencent.ig', 'com.supercell.clashofclans', 'com.king.candycrushsaga', 'com.miniclip.eightballpool',
    'com.roblox.client', 'com.zhiliaoapp.musically',
    # Travel
    'com.makemytrip', 'com.tripadvisor.tripadvisor', 'com.ixigo.train.ixitrain', 'com.google.android.apps.maps', 'com.ola.customer'
]

all_app_data = []
all_reviews_data = []

for idx, package in enumerate(app_packages):
    try:
        print(f"[{idx+1}/{len(app_packages)}] Fetching data for: {package}")
        
        app_info = app(package)
        app_info['package'] = package
        all_app_data.append(app_info)
        
        app_reviews, _ = reviews(
            package,
            lang='en',
            country='us',
            count=200
        )
        for r in app_reviews:
            r['package'] = package
            all_reviews_data.append(r)

        time.sleep(1)  

    except Exception as e:
        print(f"⚠️ Failed for {package}: {e}")
        continue

df_apps = pd.DataFrame(all_app_data)
df_reviews = pd.DataFrame(all_reviews_data)

df_apps.to_csv("apps_details_40.csv", index=False)
df_reviews.to_csv("apps_reviews_40.csv", index=False)

print("✅ Scraping done! Data saved to apps_details_40.csv and apps_reviews_40.csv")
