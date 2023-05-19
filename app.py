import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
import smtplib
from prophet.serialize import model_from_json
import datetime
from streamlit_timeline import st_timeline




st.set_page_config(page_title="TruChain", page_icon=":tada:", layout="wide")
# ---- HEADER SECTION ----
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        st.title("TruChain")
        st.write(
            "Stay Ahead of the Game with TruChain."
        )
        st.write("[Learn More >](https://www.craft.do/s/ObT9EzINTmlNed)")
    with c2:
            st.image('images/scm.jpeg')



def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Description
def info(title, text):
    with st.expander(f"{title}"):
        st.write(text)

local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding_1 = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_coding_2 = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_2cwDXD.json")




st.sidebar.header('TruChain')

st.sidebar.subheader('What you want to Predict?')
selected_model = st.sidebar.selectbox('Choose:', ('Forecasting Model', 'Anomaly Detection')) 

path = "model/forecast.csv"
print(path)
df = pd.read_csv(path)





st.sidebar.markdown('''
---
Created with ❤️ by [TruChain](https://github.com/Shivansh1203/TruChain).
''')


# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What we do")
        st.write("##")
        st.write(
            """
            Supply chain forecasting refers to the process of predicting demand, supply, or pricing for a product or a range of products in a particular industry.​

            For example, the algorithms behind a forecasting model can look at data from suppliers and customers and forecast the price of a product.​

            The algorithm can also examine external factors, such as weather or other disruptive events, to further increase the precision of the pricing forecast.​

            TruChain will help users to have clear visibility into the supply chain during peak demand seasons like Diwali and Christmas, to ensure that products were delivered to retailers on time.​
           
            """
        )
        st.write("[Our Repository >]()")
    with right_column:
        if selected_model == 'Forecasting Model':
            st_lottie(lottie_coding_1, height=300, key="coding")
        else:
            st_lottie(lottie_coding_2, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")

    if(selected_model=='Forecasting Model'):

        st.header("Our Model")
    else:
        st.header("Anomaly Detection")

    if selected_model=='Forecasting Model':

        # st_lottie(lottie_coding_1, height=300, key="coding")

        # Load the forecast data
        df = pd.read_csv("model/forecast.csv")

        # Define start and end dates
        start_date = datetime.date(2015, 1, 1)
        end_date = datetime.date(2023, 12, 31)

        # Create date input
        selected_date = st.date_input(
            "Choose a date",
            value=datetime.date(2015, 1, 1),
            min_value=start_date,
            max_value=end_date,
            key="date_input"
        )

        # Filter the forecast data for the selected date
        d = selected_date.strftime("%Y-%m-%d")
        forecast = df.loc[df['ds'] == d]

        # Display the prediction information
        if not forecast.empty:
            yhat = "{:.2f}".format(float(forecast['yhat']))
            yhat_upper = "{:.2f}".format(float(forecast['yhat_upper']))
            yhat_lower = "{:.2f}".format(float(forecast['yhat_lower']))
            prediction_year_info = "On {} the predicted supply demand is between {} and {}, with a most likely demand of {}.".format(
                d, yhat_upper, yhat_lower, yhat)
            st.write(prediction_year_info)
        else:
            st.write("No prediction available for the selected date.")
        with open('model/model.json', 'r') as fin:
            m = model_from_json(fin.read())  # Load model
        forecast = pd.read_csv('model/forecast.csv')
        fig=plot_plotly(m,forecast)
        st.plotly_chart(fig)
        polar_title = "Polar Plot"
        st.header(polar_title)
        info("Info", "This shows the trend of supply demand for the given period.")
        left_column,  center_column, right_column = st.columns(3)
        with left_column:
            " "
        
        with center_column:
            st.image('images/truchain_polarplot.png')

        with right_column:
            " "
    else:
        # st_lottie(lottie_coding_2, height=300, key="coding")
        results = pd.read_csv('model/anomaly.csv')
        fig = px.scatter(results.reset_index(), x='ds', y='y', color='anomaly', title='Anomaly’s')
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="3y", step="year", stepmode="backward"),
                    dict(count=3, label="5y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        st.plotly_chart(fig)

        timeine_title = "Major Supply Chain Hikes"
        st.header(timeine_title)
        info("Info", "The timeline highlights the major events in the predicted year regarding supply demands.")
        path = "model/anomaly.csv"
        df = pd.read_csv(path)

        items = []
        i = 1
        for index, row in df.iterrows():
            anomaly = str(row["anomaly"])
            if anomaly == "Yes":
                content = "On {}, it is expected to experience an anomaly in supply demand.".format(
                    str(row["ds"]))
                item = {"id": i, "content": "⚠",
                        "message": content, "start": str(row["ds"])}
                items.append(item)
                i += 1

        options = {
            "min": "2015-01-04",
            "max": "2017-01-03"
        }

        timeline = st_timeline(items, groups=[], options=options, height="300px")
        st.subheader("Selected item")
        st.write(timeline)





# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("Our Approach")
    text_column, image_column= st.columns((2))
    with text_column:
        st.subheader("")
        st.write(
            """
            A machine learning model will be used to predict future statistics in a supply chain using previous datasets​

            The goal is to forecast distorted demand signals in the supply chain using non-linear machine learning techniques, specifically focusing on forecasting demand at the upstream end to avoid the bullwhip effect​

            A dataset including product information, dates, locations, prices, and quantities will be used to train the model using various machine learning techniques, including linear regression, neural networks, support vector machines, and more​
            """
        )

    with image_column:
        st.image("images/epics_truchain.jpeg")

with st.container():
        st.subheader("Our Vision")
        st.write(
            """
            The proposed model can be implemented as a robust platform to aid in decision-making for the supply chain.​

            Thus, the following qualitative support from the data and concepts established in the background work is enough to prove that the proposed model can be implemented as a robust platform that can act as an aid to the client in making a better decision for the supply chain.
            """
        )
    
# ---- CONTACT ----

with st.container():
    st.write("---")
    st.header("Get In Touch With Us")
    st.write("##")

 
    def send_email(name, email, message):
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("epicstruchain@gmail.com", "imttyfmbkriojftd")
        msg = f"Subject: New message from {name}\n\n{name} ({email}) sent the following message:\n\n{message}"
        server.sendmail("epicstruchain@gmail.com", "epicstruchain@gmail.com", msg)
        st.success("Thank you for contacting us.")
        
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")

    if st.button("Send"):
        send_email(name, email, message)

    
    st.markdown(
    """
    <style>
       
         /* Adjust the width of the form elements */
        .stTextInput {
            width: 50%;
        }
        
        .stTextArea {
            width: 20%;
        }
        /* Style the submit button */
        .stButton button {
            background-color: #45a049;
            color: #FFFFFF;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            width: 10%;
        }
        /* Style the success message */
        .stSuccess {
            color: #0072C6;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


