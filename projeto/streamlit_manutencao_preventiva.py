import pandas as pd
import shap
import streamlit as st
from PIL import Image
import plotly.express as px

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor


# Dataset de treino
df_maintenance = pd.read_csv('PM_train.txt',delimiter= ' ',  header = None)
df_maintenance.drop(columns=[26,27], inplace= True)

# Dataset de teste
df_test = pd.read_csv('PM_test.txt',delimiter= ' ',  header = None)
df_test.drop(columns=[26,27], inplace= True)

# Renomeando as colunas

df_maintenance.columns= ['asset_id','runtime','setting1', 'setting2','setting3',
                         'tag1','tag2','tag3','tag4','tag5','tag6','tag7','tag8','tag9','tag10',
                         'tag11','tag12','tag13','tag14','tag15','tag16','tag17','tag18','tag19','tag20',
                         'tag21']

df_test.columns= ['asset_id','runtime','setting1', 'setting2','setting3',
                         'tag1','tag2','tag3','tag4','tag5','tag6','tag7','tag8','tag9','tag10',
                         'tag11','tag12','tag13','tag14','tag15','tag16','tag17','tag18','tag19','tag20',
                         'tag21']

# Transformando a variável asset em string
df_maintenance['asset_id'] = df_maintenance.asset_id.astype(str)

# Verificando a quantidade máxima de execução de cada ativo até sua falha
max_runtime_asset = df_maintenance.groupby('asset_id').runtime.max().rename('max_runtime').reset_index()

# Utilizando apenas os dados após o período de 128 ciclos
df_maintenance = df_maintenance[df_maintenance['runtime'] > 128]

#  Pegando o valor máximo de cada asset para calcular o RUL
df_maintenance = pd.merge(df_maintenance,max_runtime_asset, on='asset_id', how = 'left')
df_maintenance['rul'] = df_maintenance.apply(lambda x: x['max_runtime'] - x['runtime'], axis=1)

# Separando as features relevantes para o modelo
df_train = df_maintenance.drop(columns= ['asset_id','runtime','setting1','setting2','setting3',
                                         'tag1','tag5','tag6','tag10','tag16','tag18','tag19',
                                         'max_runtime'])

# Preparando os dados de teste
df_test = df_test.groupby('asset_id').last().reset_index()

X_test = df_test.drop(columns= ['asset_id','runtime','setting1','setting2','setting3',
                        'tag1','tag5','tag6','tag10','tag16','tag18','tag19'])

# Separando o valor target do dataframe
X = df_train.drop(columns='rul')
Y = df_train['rul']

# Separando os dados em valores de treino e teste
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.7, test_size=0.3,
                                                      random_state=0)

# Padronizando os valores dos sensores escolhidos para a mesma escala
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_valid)

et_reg = ExtraTreesRegressor().fit(X_train, y_train)
Y_et_reg_prediction = et_reg.predict(X_valid)


Y_prediction = et_reg.predict(X_test)
Y_prediction = list(Y_prediction)
test_asset = list(df_test.asset_id.unique())

df_predictions = pd.DataFrame({'asset_id':test_asset,'RUL':Y_prediction})
df_predictions['maintenance'] = df_predictions['RUL'].apply(lambda x: 'SIM' if (x-20) < 0 else 'NÃO')
df_predictions = df_predictions.sort_values(by='RUL')

############# STREAMLIT

image = Image.open('Capture.JPG')
col11, col22 = st.beta_columns([1, 19])
with col11:
    st.image(image, width=64)

with col22:
    st.title('Monitoramento de Ativos')

st.sidebar.write('''# Configurações''')
choice = st.sidebar.selectbox('Visualização',('Geral','Ativo'))

if choice == 'Geral':
    st.write('**Asset_id**: código do ativo')
    st.write('**RUL**: tempo de vida restante do ativo')
    st.write('**Maintenance**: se o ativo precisa de manutenção')

    col11, col22 = st.beta_columns([1, 2])
    with col11:
        st.dataframe(df_predictions.head(50),500,500)

    with col22:
        st.dataframe(df_predictions.tail(50),500,500)

elif choice == 'Ativo':
    n_asset = st.sidebar.selectbox('Asset',('61'))

    # SHAP
    explainer = shap.TreeExplainer(et_reg, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_test)
    asset = 61
    shap.force_plot(explainer.expected_value, shap_values[asset, :], X_valid.iloc[asset, :])

    # Colocando o nome dos sensores usados em uma lista
    sensor = ['tag2', 'tag3', 'tag4', 'tag7', 'tag8', 'tag9', 'tag11',
              'tag12', 'tag13', 'tag14', 'tag15', 'tag17', 'tag20', 'tag21']

    preditores = shap_values[asset, :]
    df_shap = pd.DataFrame({'sensor': sensor, 'feature_strengh': preditores})
    df_shap.sort_values(by='feature_strengh')

    # Sensores com influência positiva
    df_shap_positive = df_shap[df_shap['feature_strengh'] >= 1].copy()
    df_shap_positive = df_shap_positive.sort_values(by='feature_strengh', ascending=False)

    # Sensores com influência negativa
    df_shap_negative = df_shap[df_shap['feature_strengh'] <= -1].copy()
    df_shap_negative['feature_strengh'] = df_shap_negative['feature_strengh'].apply(lambda x: x * (-1))
    df_shap_negative = df_shap_negative.sort_values(by='feature_strengh', ascending=False)

    fig_positive = px.bar(df_shap_positive, x='sensor', y='feature_strengh')
    fig_negative = px.bar(df_shap_negative, x='sensor', y='feature_strengh')
    fig_negative.update_traces(marker_color='red')

    st.write('''# Visão Geral do sensores''')
    st.write('A seguir é mostrado saúde do ativo em relação aos sensores que estão acoplados')

    col11, col22 = st.beta_columns([1, 6])
    with col11:
        st.button('Reportar erro')

    with col22:
        st.button('Relatório')

    st.subheader('Sensores que estão indicando vida útil menor para o ativo')
    st.plotly_chart(fig_negative)

    st.subheader('Sensores que estão com valores saudáveis')
    st.plotly_chart(fig_positive)