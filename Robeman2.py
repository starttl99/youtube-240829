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
DEVELOPER_KEY = "YOUR_API_KEY_HERE"  # 여기에 YouTube API 키를 입력하세요

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

def get_channel_videos(channel_id):
    videos = []
    next_page_token = None
    
    while True:
        request = youtube.search().list(
            part="id,snippet",
            channelId=channel_id,
            maxResults=50,
            type="video",
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            videos.append(item['id']['videoId'])
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return videos

def get_video_comments(video_id):
    comments = []
    next_page_token = None
    
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return comments

def categorize_comments(comments):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(comments)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    return kmeans.labels_

def main():
    st.title("YouTube 채널 댓글 분석기")

    channel_id = st.text_input("YouTube 채널 ID를 입력하세요:")
    
    if st.button("분석 시작"):
        with st.spinner("채널의 동영상을 검색 중..."):
            video_ids = get_channel_videos(channel_id)
        
        if video_ids:
            st.write(f"{len(video_ids)}개의 동영상을 찾았습니다.")
            
            all_comments = []
            progress_bar = st.progress(0)
            
            for i, video_id in enumerate(video_ids):
                comments = get_video_comments(video_id)
                all_comments.extend(comments)
                progress_bar.progress((i + 1) / len(video_ids))
            
            st.write(f"총 {len(all_comments)}개의 댓글을 수집했습니다.")
            
            if all_comments:
                categories = categorize_comments(all_comments)
                
                df = pd.DataFrame({
                    'comment': all_comments,
                    'category': categories
                })
                
                st.write("댓글 분류 결과 (샘플):")
                st.dataframe(df.sample(min(len(df), 100)))
                
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
                st.write("댓글을 가져오는 데 실패했습니다.")
        else:
            st.write("채널에서 동영상을 찾을 수 없습니다. 채널 ID를 확인해주세요.")

if __name__ == "__main__":
    main()