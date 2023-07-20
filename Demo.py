import pandas as pd
import streamlit as st
import os
from neo4j import GraphDatabase
import torchaudio
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import matplotlib.pyplot as plot
import numpy as np
import scipy.signal as signal

st.set_page_config(page_title="Music discovery", page_icon=":music:", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h1 style='text-align: center; color: red;'>Music Discovery for the fun hearted</h1>", unsafe_allow_html=True)

GRAPH_DB_CREDS = {
    "HOST":os.environ['NEO4J_HOST'],
    "USER" :  os.environ['NEO4J_USER'],
    "PASSWORD": os.environ['NEO4J_PASSWORD']
}

audio_files_dir = 'presentation_audio/10_seconds_samples/'


def return_spectrum(waveform):
    # Plot the signal read from wav file
    plot.subplot(211)
    plot.plot(waveform)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')
    plot.subplot(212)
    plot.specgram(waveform, Fs=16000)
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    return plot


class Neo:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = self.driver.session()

    def close(self):
        self.driver.close()

    def run_read(self, fx_name, **kwargs):
        results = None
        session = self.driver.session()
        results = session.read_transaction(fx_name, **kwargs)
        return results

    def run_write(self, fx_name, **kwargs):
        results = None
        session = self.driver.session()
        results = session.write_transaction(fx_name, **kwargs)
        return results

    def run_mult(self, queries, **kwargs):
        tx = self.driver.session().begin_transaction()
        try:
            for query in queries:
                result = tx.run(query)
                print(result.data())
            tx.commit()
        except Exception as ex:
            tx.rollback()
            print("couldn't run the command")
            print(ex)
        tx.close()

    def run_query(self, query, **kwargs):
        #         data = None
        try:
            tx = self.session.begin_transaction(timeout=300)
            result = tx.run(query, **kwargs)
            data = result.to_df()
            tx.commit()
            tx.close()
            return data
        except Exception as ex:
            print("couldn't run the command")
            print(ex)
            print(query)
            return query

    def run_query_graph(self, query, **kwargs):
        #         data = None
        try:
            tx = self.session.begin_transaction(timeout=300)
            result = tx.run(query, **kwargs)
            return result, tx
        except Exception as ex:
            print("couldn't run the command")
            print(ex)
            print(query)
            return query

    def run_queries(self, queries, **kwargs):
        count = 1
        for q in queries:
            try:
                print(f"pushing company no. {count}")
                self.session.run(q)
            except Exception as ex:
                print("couldn't run the command")
                print(ex)
                print(q)
            count += 1
        print(f"Ran total queries: {len(queries)}")


db = Neo(GRAPH_DB_CREDS["HOST"],GRAPH_DB_CREDS["USER"], GRAPH_DB_CREDS["PASSWORD"])

st.markdown("### Discovering new music for a chosen sample")


def query_builder_track_similarity(track_name, model_name):
    query = '''MATCH (source_artist:artist)-[:PRODUCED_BY]-(source_song:audio)-[:HAS_SAMPLE]-(a1:audio)-[r:HAS_SIMILARITY]->(a2:audio)<-[:HAS_SAMPLE]-(sample_song:audio)-[:PRODUCED_BY]->(sample_artist:artist) 
    where a1.name = '{}' 
    and r.model = '{}' and source_artist.name <> sample_artist.name 
    and source_song.name <> sample_song.name return 
    source_artist.name as selected_artist ,
    a2.name as similar_sample,
    sample_song.name as similar_song, 
    sample_artist.name as similar_artist,
    r.similarity_score as similarity_score order by r.similarity_score desc'''.format(track_name, model_name)

    return query



def get_topk(track_name, model_name, k=1):
    query = query_builder_track_similarity(track_name=track_name, model_name=model_name)
    result = db.run_query(query)
    return result


def create_playlist(seed_track, model_name, max_number_of_tracks=10):
    track_list = []
    wv, sr = torchaudio.load(os.path.join(audio_files_dir, seed_track))
    wv = wv.squeeze().numpy()
    all_tracks = []
    all_tracks.append(seed_track)
    for i in range(max_number_of_tracks):
        result = get_topk(track_name=seed_track, model_name=model_name)
        if result.shape[0] > 1:
            seed_track = result.iloc[0]['similar_sample']
            track_list.append(result.iloc[0])
        else:
            break

    df = pd.DataFrame(track_list)
    st.write(df)
    sample_list = df['similar_sample'].tolist()
    concat_audio = None
    for sample in sample_list:
        all_tracks.append(sample)
        wv_2, sr_2 = torchaudio.load(os.path.join(audio_files_dir, sample))
        wv_2 = wv_2.squeeze().numpy()
        if concat_audio is None:
            concat_audio = np.concatenate([wv, wv_2])
        else:
            concat_audio = np.concatenate([concat_audio, wv_2])

    cypher_query = 'MATCH '
    for i, track in enumerate(all_tracks):
        cypher_query = cypher_query + '(a{}:audio where a{}.name="{}")-->[r{}:HAS_SIMILARITY where r{}.model="{}"]-->'.format(i,i,track, i,i, model_name)
        #cypher_query = cypher_query + '(a:audio where a.name="{}")-[r:HAS_SIMILARITY where r.model="{}"]-'.format(track, model_name)

    print(cypher_query)
    return concat_audio




def similarity(track_name, model_name):
    st.markdown("### Fetching similar samples using {} model for similarity".format(model_name))
    query = query_builder_track_similarity(track_name, model_name)
    result = db.run_query(query)
    gb = GridOptionsBuilder.from_dataframe(result)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True,) #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        result,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=False,
        height=350,
        width='100%',
        reload_data=True,
        key=model_name,
    )

    selected = grid_response['selected_rows']
    if len(selected) > 0:
        selected_sample = selected[0]['similar_sample']
        waveform, sample_rate = torchaudio.load(os.path.join(audio_files_dir, selected_sample))
        waveform = waveform.squeeze().numpy()
        st.audio(waveform, sample_rate=sample_rate)
        st.markdown("### Spectrum of chosen sample")
        plot = return_spectrum(waveform)
        st.pyplot(plot.show())

with st.container():
    with st.expander("Sample", expanded=True):
        filenames = []
        for files in os.listdir(audio_files_dir):
            filenames.append(os.path.join(files))

        selected_option = st.selectbox("Pick an audio sample", key="input_box", options=filenames)
        st.markdown("### Selected audio track is {}".format(selected_option))
        waveform, sample_rate = torchaudio.load(os.path.join(audio_files_dir, selected_option))
        waveform = waveform.squeeze().numpy()
        st.audio(waveform, sample_rate=sample_rate)
        st.markdown("### Spectrum of chosen sample")
        plot = return_spectrum(waveform)
        st.pyplot(plot.show())

    with st.container():
        col1, col2 = st.columns(2, )
        with col1:
            st.write(selected_option)
            with st.expander("AST similarity", expanded=True):
                similarity(track_name=selected_option, model_name='ast')

        with col2:
            st.write(selected_option)
            with st.expander("Clap similarity", expanded=True):
                similarity(track_name=selected_option, model_name='clap')



    st.markdown("### Building playlist using model {}".format('ast'))
    with st.container():
        with st.expander("Playlist exploration", expanded=True):
            playlist = create_playlist(seed_track=selected_option, model_name='ast')
            st.audio(playlist, sample_rate=sample_rate)


    st.markdown("### Building playlist using model {}".format('clap'))
    with st.container():
        with st.expander("Playlist exploration", expanded=True):
            playlist = create_playlist(seed_track=selected_option, model_name='clap')
            st.audio(playlist, sample_rate=sample_rate)

