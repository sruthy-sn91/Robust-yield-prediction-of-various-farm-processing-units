import streamlit as st
import pandas as pd
import pickle
model = pickle.load(open('model.pkl', 'rb'))


def predict(df):
    prediction = model.predict(df)
    return prediction


def main():
    st.title("")



# Farm:

html_temp = """ <div style="background-color:teal;padding:10px;margin-bottom:30px;">
    <h3 style="color:white;text-align:center;">Robust yeild prediction</h3>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
uploaded_file=st.file_uploader('Choose a CSV file')
result = ""
if st.button("Predict"):
    df = pd.read_csv(uploaded_file)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    df.ingredient_type = le.fit_transform(df.ingredient_type)
    df.farming_company = le.fit_transform(df.farming_company)
    df.deidentified_location_x = le.fit_transform(df.deidentified_location_x)
    df.new = le.fit_transform(df.new)
    df['ingredient_type'] = df['ingredient_type'].astype('category')
    # df['num_processing_plants'] = df['num_processing_plants'].astype('float')
    df['farming_company'] = df['farming_company'].astype('category')
    df['deidentified_location_x'] = df['deidentified_location_x'].astype('category')
    df['new'] = df['new'].astype('category')
    result = predict(df)
    
    # ans = 'No' if result == 0 else 'Yes'
    # st.success('The output is {}'.format(result))
    # st.success('The employee can be promoted: {}'.format(ans))
    st.write(result)

main()
