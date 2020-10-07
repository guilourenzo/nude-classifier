import streamlit as st
import pandas as pd
from PIL import Image
from tempfile import NamedTemporaryFile

from classifier import predict_single_image

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Classificador de Nudez")
st.header("Modelo criado como entregável relacionado ao Projeto Final do curso de *MBA em Ciência de Dados*")
st.write("**Desenvolvedor:** *Guilherme Lourenço*")

uploaded_file = st.file_uploader("Selecione uma Imagem!", type=["jpg", "jpeg", "png"])
temp_file = NamedTemporaryFile(delete=False)

if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file)
    st.image(image_uploaded, caption='Imagem Importada', use_column_width=True)
    st.write("")
    st.spinner("Classificando a sua imagem")

    temp_file.write(uploaded_file.getvalue())
    
    label = predict_single_image(temp_file.name)
    
    st.success("Imagem Classificada com Sucesso!")
    st.balloons()

    if label[0] > label[1]:
        st.error(f"A imagem possui  {str(round(label[0]*100, 2)) + ' %'} de probabilidade de conter conteúdo impróprio !")
    else:
        st.success(f"A imagem possui  {str(round(label[1]*100, 2)) + ' %'} de probabilidade de conter conteúdo livre !")

    st.info("Abaixo está descrito a probabilidade de Classificação da Imagem entre as Classes:")

    st.write(pd.DataFrame({
        'Imagem com conteudo Improprio': str(round(label[0]*100, 2)) + '%',
        'Imagem com conteudo Livre': str(round(label[1]*100, 2)) + '%',
    }, index=[0]))

st.write("Modelo baseado no projeto disponível em : https://github.com/GantMan/nsfw_model")
st.write("Para mais informações entre em contato: https://guilourenzo.github.io/")