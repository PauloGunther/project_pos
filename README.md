======== PREVISÃO DE PREÇOS DOS IMÓVEIS EM GOIÂNIA =========

Aqui se encontram os dados e os notebooks usados para realização do trabalho de Conclusão de Curso 
de MBA em Data Science Analytics, na USP ESALQ.

O primeiro notebook é o tcc_etl.ipynb - Onde está a compilação dos dados e ajustes das variaveis
O segundo é o tcc_modelos.ipynb - Onde estão os tres modelos trabalhados
Gridsearch.py é o arquivo com funções que usei para encontrar os melhores modelos
Fipezap.xlsx são os preços dos imóveis
df_pre_model.pkl é a saída dos dados pré processados do primeiro notebook.


O mercado imobiliário de Goiânia apresentou expressiva valorização nos últimos anos, destacando-se nacionalmente 
em lançamentos e demanda, porém inserido em um cenário macroeconômico desafiador. Este trabalho teve como objetivo 
prever o comportamento do preço do metro quadrado residencial na cidade e identificar variáveis macroeconômicas 
determinantes em sua formação. Para isso, foram utilizados dados da FIPEZAP e do Banco Central de janeiro de 2014 
a junho de 2025. A metodologia envolveu a aplicação de três modelos de previsão: SARIMAX, XGBoost e Redes Neurais 
Recorrentes (LSTM), utilizando variáveis defasadas para evitar data leakage. Os resultados indicaram que o saldo 
de crédito imobiliário, o índice de commodities agrícolas, a taxa de juros e o IPCA foram as variáveis mais 
relevantes. Na comparação dos modelos, o XGBoost obteve o menor erro de previsão no teste, enquanto o LSTM 
apresentou o melhor equilíbrio entre treino e teste. Conclui-se que técnicas de machine learning mostraram-se 
eficazes na previsão de preços, oferecendo insights valiosos para investidores e formuladores de políticas.


Paulo Gunther
