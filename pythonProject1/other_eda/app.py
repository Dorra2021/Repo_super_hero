import streamlit as st
import pandas as pd
import altair as alt
import numpy

st.set_page_config(
    page_title="Super Hereos Viz app",
    page_icon="marvel-vs-dc.jpg",

    initial_sidebar_state="expanded",
)
st.sidebar.image("im.png", width=280)
st.sidebar.title('EDA dashboard by LYD-FR')
# Load the dataset
df = pd.read_csv('superheroes_dataset.csv')
df = df.dropna(subset=['creator']).reset_index(drop=True)
conditions = [
        df.creator.str.contains("Marvel"),
        df.creator.str.contains("DC")
    ]
choices = ["Marvel", "DC"]
df["creator"] = numpy.select(conditions, choices, default="Other")
df = df.dropna(subset=['alignment']).reset_index(drop=True)
df.gender.fillna("No info", inplace=True)

# Create a filter for selecting the publisher
publisher = st.sidebar.selectbox('Select Publisher', ["Comparative study",'Marvel', 'DC'])
print(publisher)
# Filter the dataset by publisher
if publisher != "Comparative study":
    filtered_df = df.loc[df.creator.str.contains(publisher)]
else:
    filtered_df = df

# Create a filter for selecting the character
characters = ["All", "alignment", "power"]#filtered_df['name'].unique()
feature = st.sidebar.selectbox('Select feature to study', characters)

# Filter the dataset by character
#filtered_df = filtered_df[filtered_df['name'] == character]

# Display the filtered dataset
cols_quanti = ['overall_score', 'intelligence_score',
              'strength_score', 'speed_score',
              'durability_score', 'power_score',
              'combat_score',]
st.write( "#### <font color='purple'> ***Descriptive statistics.*** </font>",unsafe_allow_html=True )
st.markdown( "### \t This is the main stats for the selected publisher.",unsafe_allow_html=True )
st.write(filtered_df[cols_quanti].describe().T)

# Create a histogram for the appearances of the character

hist_chart = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X('alignment', bin=False),
    y='count()',
    color='creator',
    tooltip=['alignment', 'count()']
).interactive()

hist_chart_bis = alt.Chart(filtered_df).mark_bar().encode(
    alt.Column('alignment'), alt.X('creator'),
    alt.Y('count()', axis=alt.Axis(grid=False)),
    alt.Color('creator'))

# Display the histogram
st.altair_chart(hist_chart_bis, use_container_width=False)
st.altair_chart(hist_chart, use_container_width=True)


hist_chart_bis = alt.Chart(filtered_df).mark_bar().encode(
    alt.Column('gender'), alt.X('creator'),
    alt.Y('count()', axis=alt.Axis(grid=False)),
    alt.Color('creator'))
st.altair_chart(hist_chart_bis, use_container_width=False)