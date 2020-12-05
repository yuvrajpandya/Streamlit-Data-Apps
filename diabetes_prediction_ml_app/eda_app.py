import streamlit as st 
import pandas as pd 

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px 


@st.cache
def load_data(data):
	df = pd.read_csv(data)
	return df


def run_eda_app():
	st.subheader("EDA Section")
	df = load_data("data/diabetes_data_upload.csv")
	df_clean = load_data("data/diabetes_data_upload_encoded.csv")
	freq_df = load_data("data/freqdist_of_age_data.csv")

	submenu = st.sidebar.selectbox("SubMenu",["Descriptive","Plots"])
	if submenu == "Descriptive":
		
		st.dataframe(df)

		with st.beta_expander("Data Types Summary"):
			st.dataframe(df.dtypes)

		with st.beta_expander("Descriptive Summary"):
			st.dataframe(df_clean.describe())

		with st.beta_expander("Gender Distribution"):
			st.dataframe(df['Gender'].value_counts())

		with st.beta_expander("Class Distribution"):
			st.dataframe(df['class'].value_counts())
	else:
		st.subheader("Plots")

		# Layouts
		col1,col2 = st.beta_columns([2,1])
		with col1:
			with st.beta_expander("Dist Plot of Gender"):

				gen_df = df['Gender'].value_counts().to_frame()
				gen_df = gen_df.reset_index()
				gen_df.columns = ['Gender Type','Counts']

				p01 = px.pie(gen_df,names='Gender Type',values='Counts')
				st.plotly_chart(p01,use_container_width=True)

			with st.beta_expander("Dist Plot of Class"):
				fig = plt.figure()
				sns.countplot(df['class'])
				st.pyplot(fig)

			with st.beta_expander("Gender Distribution across Classes"):
				# st.dataframe(df['class'].value_counts())
				gender_classes = df.groupby(['Gender', 'class'])['Age'].count().reset_index().rename(columns={'Age':'Count', 'class':'Class'})

				fig = px.sunburst(gender_classes, path=['Gender', 'Class'], values='Count', color='Class',
                 color_discrete_map={'Positive':'red', 'Negative':'lightgreen'})
				st.plotly_chart(fig, use_container_width=True)

		with col2:
			with st.beta_expander("Gender Distribution"):
				st.dataframe(df['Gender'].value_counts())


		with st.beta_expander("Frequency Dist Plot of Age"):

			p = px.bar(freq_df,x='Age',y='count')
			st.plotly_chart(p)

			p2 = px.line(freq_df,x='Age',y='count')
			st.plotly_chart(p2)

		with st.beta_expander("Outlier Detection Plot"):
			fig = plt.figure()

			p3 = px.box(df,x='Age',color='Gender')
			st.plotly_chart(p3)

		with st.beta_expander("Correlation Plot"):
			corr_matrix = df_clean.corr()
			fig = plt.figure(figsize=(20,10))
			sns.heatmap(corr_matrix,annot=True)
			st.pyplot(fig)