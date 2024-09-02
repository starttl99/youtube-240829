import streamlit as st
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import datetime

# YouTube API 설정
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = ""  # 여기에 YouTube API 키를 입력하세요

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

def get_video_comments(video_id):
    comments = []
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    ).execute()

    while results:
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        if "nextPageToken" in results:
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                pageToken=results["nextPageToken"],
                maxResults=100
            ).execute()
        else:
            break

    return comments

def categorize_comments(comments):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(comments)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    return kmeans.labels_

def main():
    st.title("YouTube 댓글 분석기")

    video_id = st.text_input("V3pbFkJ-jXM&t")
    
    if st.button("분석 시작"):
        comments = get_video_comments(video_id)
        
        if comments:
            categories = categorize_comments(comments)
            
            df = pd.DataFrame({
                'comment': comments,
                'category': categories
            })
            
            st.write("댓글 분류 결과:")
            st.dataframe(df)
            
            category_counts = Counter(categories)
            
            fig, ax = plt.subplots()
            ax.bar(category_counts.keys(), category_counts.values())
            ax.set_xlabel("카테고리")
            ax.set_ylabel("댓글 수")
            ax.set_title("카테고리별 댓글 분포")
            st.pyplot(fig)
            
            # 시간에 따른 변화 추적을 위한 데이터 저장
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_data = st.session_state.get('history_data', [])
            history_data.append((current_time, category_counts))
            st.session_state['history_data'] = history_data
            
            if len(history_data) > 1:
                st.write("시간에 따른 카테고리 변화:")
                fig, ax = plt.subplots()
                for i in range(5):
                    values = [data[1][i] for data in history_data]
                    ax.plot([data[0] for data in history_data], values, label=f'카테고리 {i}')
                ax.set_xlabel("시간")
                ax.set_ylabel("댓글 수")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.write("댓글을 가져오는 데 실패했습니다. 동영상 ID를 확인해주세요.")

if __name__ == "__main__":
    main()
