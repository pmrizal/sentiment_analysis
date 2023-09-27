import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

import nltk
nltk.download('punkt')

from google_play_scraper import Sort, reviews_all

import text_analysis as nlp

model = nlp.Model()

def main():		
    activities = ["Google Play Scraper","Bulk","Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Choice",activities)

    if choice == 'Google Play Scraper':
        st.title('Sentiment Analysis App')
        st.markdown('This application is all about sentiment analysis')

        st.subheader('Scrap Data from Google Play Store Review')
        result = reviews_all(
            'co.talenta',
            sleep_milliseconds=0, # defaults to 0
            lang='id',            # defaults to 'en'
            country='id',         # defaults to 'us'
            sort=Sort.NEWEST,     # defaults to Sort.MOST_RELEVANT , you can use Sort.NEWEST to get newst reviews
        )
        dfid = pd.DataFrame(np.array(result),columns=['review'])
        dfid = dfid.join(pd.DataFrame(dfid.pop('review').tolist()))
        dfid['date'] = pd.to_datetime(dfid['at'],format='%Y%m%d')
        
        col1, col2 = st.columns(2)
        with col1:
            d = st.date_input("Start date", datetime.date(2022,1,1))
        with col2:
            f = st.date_input("End date", datetime.date(2022,2,1))
        d = d.strftime('%Y-%m-%d')
        f = f.strftime('%Y-%m-%d')
        dfid_filter = dfid.loc[((dfid['date'] >= d) & (dfid['date'] <= f))]

        dfthumbs = dfid_filter.sort_values(by='thumbsUpCount', ascending=False)
        dfthumbs = dfthumbs[['userName','content','score','thumbsUpCount','at']]
        dfthumbs5 = dfthumbs.head(10)

        dfpre = dfid_filter[['content','score']]
        dfpre['filtered'] = dfpre['content'].apply(nlp.filtering_text)
        dfpre['cleaned'] = dfpre['filtered'].apply(nlp.stop_stem)
        dfpre['tokens'] = dfpre['cleaned'].apply(nlp.word_tokenize_wrapper)        
        dfpre['normalisasi'] = dfpre['tokens'].apply(nlp.normalisasi_kata)

        freq_df = nlp.count_words(dfpre)
        idf_df = nlp.compute_idf(dfpre)
        freq_df['tfidf'] = freq_df['freq'] * idf_df['idf']
        
        #if st.button("Show Scrap Result"):
        st.markdown('Scrap Result')
        st.dataframe(dfid_filter)

        st.markdown('Top like review')
        st.dataframe(dfthumbs5)
        
        #if st.button("Show Preprocess Result"):
        st.markdown('Preprocess Result')
        st.dataframe(dfpre)

        #if st.button("Show EDA"):
        st.markdown('EDA Result')
            
        df_freq = freq_df.reset_index()

        fig7 = px.histogram(dfid_filter, x='score', title='Total setiap rating')
        fig7.update_layout(showlegend=False)
        fig7.update_layout(bargap=0.1)
        st.plotly_chart(fig7)

        topkn = st.slider("Top k most common words", min_value=10, max_value=25, step=5, value=10)
        df_word_dist = df_freq.head(topkn)
        fig5 = px.bar(df_word_dist, x="token", y="freq", title='Distribusi Kata')
        fig5.update_layout(showlegend=False)
        fig5.update_xaxes(tickangle=-90)
        st.plotly_chart(fig5)

        st.markdown('N-gram Analysis')
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("N for the N-gram", min_value=2, max_value=5, step=1, value=2)
        with col2:
            topk = st.slider("Top k most common phrases", min_value=10, max_value=25, step=5, value=10)
        dfpre['bigrams'] = dfpre['normalisasi'].apply(nlp.ngrams, n=n)
        df2 = nlp.count_words(dfpre, 'bigrams').head(topk)
        df3 = df2.reset_index()

        fig6 = px.bar(df3, x='token', y='freq',title="N-gram Analysis")
        fig6.update_xaxes(tickangle=-90)
        st.plotly_chart(fig6)

        st.markdown('Summary')
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Rating", round(dfid_filter.score.mean(),2))
        col2.metric("Top Word", df_word_dist['token'][0])
        col3.metric("Top N-gram", df3['token'][0])

        st.markdown('Download Scrap Result')
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label="Download data as CSV", data=nlp.convert_df_to_csv(dfid_filter), file_name='dfid_filter.csv', mime='text/csv')
        with col2:
            st.download_button(label="Download all data as CSV", data=nlp.convert_df_to_csv(dfid), file_name='dfid_all.csv', mime='text/csv',)
        
        
    
    if choice == 'Bulk':
        st.title('Sentiment Analysis App')
        st.markdown('This application is all about sentiment analysis')
        st.subheader("Text Analysis on Bulk File")
        uploaded_file2 = st.file_uploader("Choose a file")
        if uploaded_file2 is not None:
            dfbulk = pd.read_csv(uploaded_file2)
            st.write('Upload Successful')
        
            dfbulk['filtered'] = dfbulk['content'].apply(nlp.filtering_text)
            dfbulk['cleaned'] = dfbulk['filtered'].apply(nlp.stop_stem)
            dfbulk['tokens'] = dfbulk['cleaned'].apply(nlp.word_tokenize_wrapper)        
            dfbulk['normalisasi'] = dfbulk['tokens'].apply(nlp.normalisasi_kata)

            st.markdown('Preprocess Result')
            st.dataframe(dfbulk)

            st.markdown('EDA Result')
            
            freq_df2 = nlp.count_words(dfbulk)
            idf_df2 = nlp.compute_idf(dfbulk)
            freq_df2['tfidf'] = freq_df2['freq'] * idf_df2['idf']
            df_freq2 = freq_df2.reset_index()

            topkn2 = st.slider("Top k most common words", min_value=10, max_value=25, step=5, value=10)
            df_word_dist2 = df_freq2.head(topkn2)
            fig8 = px.bar(df_word_dist2, x="token", y="freq", title='Distribusi Kata')
            fig8.update_layout(showlegend=False)
            fig8.update_xaxes(tickangle=-90)
            st.plotly_chart(fig8)            

            st.markdown('N-gram Analysis')
            col1, col2 = st.columns(2)
            with col1:
                n = st.slider("N for the N-gram", min_value=2, max_value=5, step=1, value=2)
            with col2:
                topk2 = st.slider("Top k most common phrases", min_value=10, max_value=25, step=5, value=10)
            dfbulk['bigrams'] = dfbulk['normalisasi'].apply(nlp.ngrams, n=n)
            dfbulk2 = nlp.count_words(dfbulk, 'bigrams').head(topk2)
            dfbulk3 = dfbulk2.reset_index()

            fig9 = px.bar(dfbulk3, x='token', y='freq',title="N-gram Analysis")
            fig9.update_xaxes(tickangle=-90)
            st.plotly_chart(fig9)

            st.markdown('Summary')
            col1, col2 = st.columns(2)
            col1.metric("Top Word", df_word_dist2['token'][0])
            col2.metric("Top N-gram", dfbulk3['token'][0])

    if choice == 'Sentiment Analysis':
        st.title('Sentiment Analysis App')
        st.markdown('This application is all about sentiment analysis')

        st.subheader("Sentiment Analysis")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            dfsentiment = pd.read_csv(uploaded_file)
            st.write('Upload Successful')
            dfs = dfsentiment['content'].apply(nlp.filtering_text) 
        
            l = []
            for i in range(len(dfs)):
                review_text = dfs[i]
                sentiment = model.predict(review_text)
                l.append({"text" : review_text, "sentiment":sentiment})
        
            dfsnew = pd.DataFrame(l)
            st.write(dfsnew)
            
            dfsentiment2=dfsnew.groupby('sentiment').count().reset_index()

            fig = px.bar(dfsentiment2, x='sentiment', y='text',title="Sentiment Summary")
            fig.update_xaxes(tickangle=0)
            st.plotly_chart(fig)

            dfsent = dfsnew
            dfsent['filtered'] = dfsent['text'].apply(nlp.filtering_text)
            dfsent['cleaned'] = dfsent['filtered'].apply(nlp.stop_stem)
            dfsent['tokens'] = dfsent['cleaned'].apply(nlp.word_tokenize_wrapper)        
            dfsent['normalisasi'] = dfsent['tokens'].apply(nlp.normalisasi_kata)

            dfsentpos = dfsent.loc[(dfsent['sentiment']=='positive')]
            dfsentneg = dfsent.loc[(dfsent['sentiment']=='negative')]
            #dfsentpos = dfsent.where(pos, inplace = True)
            #dfsentneg = dfsent.where(neg, inplace = True)

            st.markdown('Words Distribution')
            
            freq_dfpos = nlp.count_words(dfsentpos)
            idf_dfpos = nlp.compute_idf(dfsentpos)
            freq_dfpos['tfidf'] = freq_dfpos['freq'] * idf_dfpos['idf']
            df_freqpos = freq_dfpos.reset_index()

            topknpos = st.slider("Top k most common words (positive)", min_value=10, max_value=25, step=5, value=10)
            df_word_distpos = df_freqpos.head(topknpos)
            figpos = px.bar(df_word_distpos, x="token", y="freq", title='Distribusi Kata positif')
            figpos.update_layout(showlegend=False)
            figpos.update_xaxes(tickangle=-90)
            st.plotly_chart(figpos)

            freq_dfneg = nlp.count_words(dfsentneg)
            idf_dfneg = nlp.compute_idf(dfsentneg)
            freq_dfneg['tfidf'] = freq_dfneg['freq'] * idf_dfneg['idf']
            df_freqneg = freq_dfneg.reset_index()

            topknneg = st.slider("Top k most common words (negative)", min_value=10, max_value=25, step=5, value=10)
            df_word_distneg = df_freqneg.head(topknneg)
            figneg = px.bar(df_word_distneg, x="token", y="freq", title='Distribusi Kata negatif')
            figneg.update_layout(showlegend=False)
            figneg.update_xaxes(tickangle=-90)
            st.plotly_chart(figneg)  

            if st.download_button(label="Download data as CSV", data=nlp.convert_df_to_csv(dfsnew), file_name='large_df.csv', mime='text/csv',):
                st.write('download successful')
        
        st.markdown('This is the end of the page')
    
    if choice == 'About':
        st.subheader("About: Sentiment Analysis App")
        st.info("Built with Streamlit and Fine Tuning Bert")
        st.text("Author: Mochamad Rizal Prasetyo")

if __name__ == '__main__':
    main()
