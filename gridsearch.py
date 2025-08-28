# Pacotes

import itertools
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX # Modelo
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error # Métricas (RMSE)
import pandas as pd


# Criando função para realizar Grid Search com SARIMAX - Melhor RMSE

def sarimax_gridsearch_rmse(y_train, y_test, exog_train=None, exog_test=None,
                           p_values=range(1, 3), d_values=range(0, 2), q_values=range(1, 3),
                           P_values=range(1, 5), D_values=range(1, 5), Q_values=range(1, 5),
                           seasonal_period=12):
    """
    Realiza grid search para SARIMAX selecionando pelo menor RMSE

    Parâmetros:
    - y_train: preços de treino
    - y_test: preços para o cálculo do RMSE
    - exog_train: variáveis exógenas de treino
    - exog_test: variáveis exógenas para teste
    - p_values, d_values, q_values: parâmetros não sazonais
    - P_values, D_values, Q_values: parâmetros sazonais
    - seasonal_period: periodicidade sazonal

    Retorna:
    - DataFrame com resultados ordenados por RMSE
    """
    # Combinação dos parâmetros não sazonais
    order_combinations = list(itertools.product(p_values,
                                                d_values,
                                                q_values))
    # Combinação dos parâmetros sazonais
    seasonal_combinations = list(itertools.product(P_values,
                                                   D_values,
                                                   Q_values,
                                                    [seasonal_period]))

    # Variaveis vazias
    results = []
    best_rmse = np.inf

    # Combinando cada parâmetro
    for order in order_combinations:
        for seasonal_order in seasonal_combinations:
          try:
                # Treinar modelo
                model = SARIMAX(y_train,
                              exog=exog_train,
                              order=order,
                              seasonal_order=seasonal_order)

                fitted_model = model.fit(disp=-1) # Silencia msgs de saída

                # Fazer previsões
                forecast = fitted_model.get_forecast(steps=len(y_test),
                                                     exog=exog_test)

                predictions = forecast.predicted_mean

                # Retornando os valores para os originais
                predictions = y.iloc[-len(predictions)-1:-1].values + predictions.cumsum()
                y_test = y.iloc[-len(predictions):]

                # Calcular RMSE
                rmse = sqrt(mean_squared_error(y_test, predictions))

                # Armazenar resultados
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'rmse': rmse,
                    'log_likelihood': fitted_model.llf,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                })

                # Atualizar melhor modelo
                if rmse < best_rmse:
                    best_rmse = rmse

                print(f"""SARIMAX{order}x{seasonal_order} - RMSE: {rmse:.2f} - AIC: {fitted_model.aic:.2f} - LL: {fitted_model.llf:.2f}""")

          # Ignora erros de combinações
          except Exception as e:
                print(f"Falha em {order}x{seasonal_order}: {str(e)}")
                continue

    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse').reset_index(drop=True)

    print("\nTop 5 melhores modelos por RMSE:")
    print(results_df.head(5).to_string(index=False))

    return results_df


# Criando função para realizar Grid Search com SARIMAX - Melhor AIC

def sarimax_gridsearch_AIC(y_train, y_test, exog_train=None, exog_test=None,
                           p_values=range(1, 3), d_values=range(0, 2), q_values=range(1, 3),
                           P_values=range(1, 5), D_values=range(1, 5), Q_values=range(1, 5),
                           seasonal_period=12):
    """
    Realiza grid search para SARIMAX selecionando pelo menor AIC

    Parâmetros:
    - y_train: preços de treino
    - y_test: preços para o cálculo do AIC
    - exog_train: variáveis exógenas de treino
    - exog_test: variáveis exógenas para teste
    - p_values, d_values, q_values: parâmetros não sazonais
    - P_values, D_values, Q_values: parâmetros sazonais
    - seasonal_period: periodicidade sazonal

    Retorna:
    - DataFrame com resultados ordenados por AIC
    """
                             
    # Combinação dos parâmetros não sazonais
    order_combinations = list(itertools.product(p_values,
                                                d_values,
                                                q_values))
    # Combinação dos parâmetros sazonais
    seasonal_combinations = list(itertools.product(P_values,
                                                   D_values,
                                                   Q_values,
                                                    [seasonal_period]))

    # Variaveis vazias
    results = []
    best_AIC = np.inf

    # Combinando cada parâmetro
    for order in order_combinations:
        for seasonal_order in seasonal_combinations:
          try:
                # Treinar modelo
                model = SARIMAX(y_train,
                              exog=exog_train,
                              order=order,
                              seasonal_order=seasonal_order)

                fitted_model = model.fit(disp=-1) # Silencia msgs de saída

                # Fazer previsões
                forecast = fitted_model.get_forecast(steps=len(y_test),
                                                     exog=exog_test)

                predictions = forecast.predicted_mean

                # Retornando os valores para os originais
                predictions = y.iloc[-len(predictions)-1:-1].values + predictions.cumsum()
                y_test = y.iloc[-len(predictions):]

                # Calcular RMSE
                rmse = sqrt(mean_squared_error(y_test, predictions))

                # Armazenar resultados
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'rmse': rmse,
                    'log_likelihood': fitted_model.llf,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                })

                # Atualizar melhor modelo
                if fitted_model.aic < best_AIC:
                    best_AIC = fitted_model.aic

                print(f"""SARIMAX{order}x{seasonal_order} - RMSE: {rmse:.2f} - AIC: {fitted_model.aic:.2f} - LL: {fitted_model.llf:.2f}""")

          # Ignora erros de combinações
          except Exception as e:
                print(f"Falha em {order}x{seasonal_order}: {str(e)}")
                continue

    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic').reset_index(drop=True)

    print("\nTop 5 melhores modelos por AIC:")
    print(results_df.head(5).to_string(index=False))


    return results_df
                            
