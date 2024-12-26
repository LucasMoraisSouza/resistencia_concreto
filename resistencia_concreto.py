# importando as bibliotecas
import pandas as pd
import  seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import streamlit as st


st.title('PREVISÃO DA RESISTÊNCIA DO CONCRETO')
st.write('---')
st.write('***O concreto é uma das soluções mais versáteis e amplamente utilizadas na construção civil, devido à sua resistência, durabilidade e capacidade de moldagem em diferentes formas.***')
st.write('***Este aplicativo utiliza inteligência artificial para prever a resistência do concreto com base em variáveis-chave, permitindo maior precisão e eficiência na análise de projetos.***') 
st.write('***Compilamos alguns componentes que costumam ser usados na mistura do concreto.***')
st.write('***Estes componentes se encontram na coluna à esquerda da tela.***')
st.write('***Pedimos para que faça o preenchimento destas variáveis para que nossa inteligência tenha condições de calcular a resistência de sua mistura.***')
st.write('---')

st.sidebar.write('**Abaixo estão os campos onde você deverá preencher com os valores escolhidos para a mistura final do seu concreto.**')
st.sidebar.write('**As unidades de medida de cada componente estão especificadas em quilogramas em cada metro cúbico de concreto (KG/m³ de concreto) exceto a idade que está disposta em dias.**')
st.sidebar.write('---')

#carregando os dados de treinamento
df = pd.read_excel('Concrete_Data.xls')
# criando nova lista para as variaveis
nomes_colunas = ['cimento', 'escória_de_alto_forno', 'cinzas_volantes', 'agua', 'superplastificante', 'agregado_graudo', 'agregado_miudo', 'idade', 'resistencia']
# aplicando os nomes às colunas
df.columns = nomes_colunas

# retirando os outliers através do metodo do intervalor interquartil
# o primeiro quartil é o valor abaixo do qual 25% dos dados estão localizados.
Q1 = df.quantile(0.25)
# o terceiro quartil é o valor abaixo do qual 75% dos dados estão localizados.
Q3 = df.quantile(0.75)
# o intervalo interquartil (IQR) é a região onde estão os dados do terceiro quartil com exceção dos dados do primeiro
IQR = Q3 - Q1

# Identificando os outliers:
# armazenando em um objeto os dados que se encontram além dos limites dos interquartis com uma margem de tolerancia igual a 1,5 X IQR
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
# armazenando num objeto todas observações que existem dentro do objeto outliers
data_outliers = df[outliers.any(axis=1)]
# criando um dataframe do pandas para trazer as observações numa tabela
df_outliers = pd.DataFrame(data_outliers)

# criando um novo dataframe que possuirá as observações cujos outliers são ausentes
df_concreto = df.drop(df_outliers.index)
# resetando os indexes do dataframe
df_concreto.reset_index(inplace=True)
# dropando o index do dataframe
df_concreto.drop(columns = 'index', inplace = True)

# dividindo os dados em variaveis explicativas e variavel alvo:
# variáveis explicativas
X = df_concreto.drop(columns = 'resistencia')
# variável alvo (target) y
y = df_concreto['resistencia']

# aplicando o train_test_split para realizar a divisão, sendo que 70% dos dados serão separados para treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 369)

# criando o modelo
# vamos chamar este modelo de gbr_model (gradient boosting regressor) e vamos armazenar nesta variavel a calculadora do gradient boosting regressor
gbr_model = GradientBoostingRegressor(n_estimators = 300, learning_rate = 0.1, max_depth = 14, min_samples_split = 2, min_samples_leaf= 20, random_state = 369)
# agora vamos treinar este modelo com os dados de teste
gbr_model.fit(X_train,y_train)


# criação dos campos de preenchimento de variáveis
inputs = {}
st.sidebar.title("Entrada de Variáveis")
# Lista de grandezas
grandezas = {
    "idade": "em dias",
    # Outras colunas com "em kg/m³"
    "variavel1": "em kg/m³",
    "variavel2": "em kg/m³",
    "variavel3": "em kg/m³",
    "variavel4": "em kg/m³",
    "variavel5": "em kg/m³",
    "variavel6": "em kg/m³",
    "variavel7": "em kg/m³"
    }

# Iteração para criar campos com as grandezas
for column in X.columns:
    unidade = grandezas.get(column, "em kg/m³")  # Padrão para kg/m³
    value = st.sidebar.number_input(
        f"Insira o valor para {column} ({unidade}):", min_value=0.0, step=0.01
    )
    inputs[column] = value

st.sidebar.write('---')

# criando um botão para gerar o cálculo da resistência
st.sidebar.write('**Clique no botão abaixo para calcular a resistência do seu concreto ↓**')
botao = st.sidebar.button('**CALCULAR A RESISTÊNCIA DO MEU CONCRETO EM MEGAPASCAL (MPa)**', key='botao_calcular')


# realizando a Previsão da resistência
if botao:
    input_values = np.array(list(inputs.values())).reshape(1, -1)
    prediction = gbr_model.predict(input_values)
    st.write("## Muito bem! ##")
    st.write("#### Com base nos valores que foram escolhidos por você na lateral da tela, nosso algorítimo foi capaz de calcular a resistência do concreto medida em MPa ####")
    st.write(f"# A resistência do seu concreto foi de {prediction[0]:.2f} MPa #")

    st.write('---')
    st.write('## Observaçõs importantes: ##')
    st.write('Lembre-se sempre: Ao realizar a concretagem, crie vários corpos de prova com o mesmo concreto para testar sua resistência em laboratório.')
    st.write('Os dados usados para o treinamento da nossa inteligência foram coletados em laboratório que é um ambiente controlado e estável.')
    st.write('Por isso a importância dos corpos de prova, pois no momento da execução podemos ser expostos a variáveis que não estão listadas na base, como temperaturas elevadas, frio intenso, chuvas ou contratempos de execução.')
    st.write('---')

    