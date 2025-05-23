import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import make_pipeline
import joblib
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
import io
import os
from scipy.optimize import minimize
from scipy.optimize import Bounds

# Инициализация session_state
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False
if 'regression_results' not in st.session_state:
    st.session_state.regression_results = {}


def load_data():
    st.subheader("Информация о файле")
    st.write("""
    Пожалуйста, загрузите файл в формате Excel (.xlsx). 
    Файл должен содержать следующие столбцы:
    - **A, B, C, D, E, F, G**: Признаки (независимые переменные).
    - **H**: Целевая переменная.
    """)

    uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("Данные успешно загружены!")
            return df
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
    return None


def show_correlation_heatmaps(df):
    with st.expander("Тепловые карты корреляций"):
        st.subheader("Тепловая карта корреляции Пирсона")
        pearson_corr_matrix = df.corr(method="pearson").round(3)  # Округление до 3 знаков
        fig_pearson = px.imshow(
            pearson_corr_matrix,
            text_auto=True,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_pearson)

        st.subheader("Тепловая карта корреляции Спирмана")
        spearman_corr_matrix = df.corr(method="spearman").round(3)  # Округление до 3 знаков
        fig_spearman = px.imshow(
            spearman_corr_matrix,
            text_auto=True,
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig_spearman)


def show_formula(coefficients, intercept, feature_names, regression_type):
    if regression_type == "Линейная":
        formula = f"H = {intercept:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" + {coef:.7f}*{feature_names[i]}"
    elif regression_type in ["Квадратическая", "Кубическая"]:
        formula = f"H = {intercept:.7f}"
        for coef, name in zip(coefficients, feature_names):
            formula += f" + {coef:.7f}*{name}"
    elif regression_type == "Логарифмическая":
        formula = f"H = {intercept:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" + {coef:.7f}*log({feature_names[i]})"
    elif regression_type == "Lasso":
        formula = f"H = {intercept:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" + {coef:.7f}*{feature_names[i]}"
    elif regression_type == "Экспоненциальная":
        a = np.exp(intercept)
        formula = f"H = {a:.7f} * exp("
        for i, coef in enumerate(coefficients):
            formula += f"{coef:.7f}*{feature_names[i]} + "
        formula = formula.rstrip(" + ") + ")"
    elif regression_type == "Степенная":
        a = np.exp(intercept)
        formula = f"H = {a:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" * {feature_names[i]}^{coef:.7f}"
    st.subheader("Формула модели")
    st.write(formula)


def show_feature_importance(coefficients, feature_names):
    st.subheader("График значимости факторов")
    importance = np.abs(coefficients)
    fig = px.bar(x=feature_names, y=importance, labels={"x": "Факторы", "y": "Важность"})
    st.plotly_chart(fig)


def run_optimization(model, X_train, y_train, regression_type, optimization_type, bounds=None):
    try:
        st.subheader("Результаты оптимизации")

        def objective(x):
            x = x.reshape(1, -1)
            if regression_type == "Экспоненциальная":
                pred = np.exp(model.predict(x)[0])
            elif regression_type == "Степенная":
                pred = np.exp(model.predict(np.log(x))[0])
            else:
                pred = model.predict(x)[0]

            return -pred if optimization_type == "Максимизация" else pred

        x0 = X_train.mean().values

        if bounds is None:
            bounds = Bounds(X_train.min().values, X_train.max().values)

        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        if result.success:
            optimal_x = result.x
            if regression_type == "Экспоненциальная":
                optimal_y = np.exp(model.predict(optimal_x.reshape(1, -1))[0])
            elif regression_type == "Степенная":
                optimal_y = np.exp(model.predict(np.log(optimal_x.reshape(1, -1)))[0])
            else:
                optimal_y = model.predict(optimal_x.reshape(1, -1))[0]

            st.success("Оптимизация завершена успешно!")
            st.write(f"Оптимальные значения факторов ({optimization_type} H):")

            optimal_df = pd.DataFrame([optimal_x], columns=X_train.columns)
            st.dataframe(optimal_df.style.format("{:.4f}"))

            st.write(f"Оптимальное значение H: {optimal_y:.4f}")

            if len(X_train.columns) <= 2:
                st.subheader("Визуализация оптимума")
                if len(X_train.columns) == 1:
                    fig = px.scatter(x=X_train.iloc[:, 0], y=y_train)
                    fig.add_vline(x=optimal_x[0], line_color="red")
                    st.plotly_chart(fig)
                else:
                    fig = px.scatter(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], color=y_train)
                    fig.add_trace(go.Scatter(x=[optimal_x[0]], y=[optimal_x[1]],
                                             mode="markers", marker=dict(color="red", size=12),
                                             name="Оптимальная точка"))
                    st.plotly_chart(fig)
        else:
            st.error(f"Ошибка оптимизации: {result.message}")

    except Exception as e:
        st.error(f"Ошибка при выполнении оптимизации: {e}")


def run_regression(df, regression_type):
    try:
        if df.isnull().any().any():
            raise ValueError(
                "В данных есть пропущенные значения (NaN). Пожалуйста, заполните их перед запуском модели.")

        X = df[["A", "B", "C", "D", "E", "F", "G"]]
        y = df["H"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if regression_type == "Линейная":
            model = make_pipeline(StandardScaler(), LinearRegression())

        elif regression_type == "Квадратическая":
            model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())

        elif regression_type == "Кубическая":
            model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())

        elif regression_type == "Логарифмическая":
            if (X_train.values <= 0).any() or (X_test.values <= 0).any():
                st.error("Логарифмическая регрессия требует положительных значений в данных.")
                return
            log_x_train = np.log(X_train)
            log_x_test = np.log(X_test)
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(log_x_train, y_train)
            y_pred = model.predict(log_x_test)

        elif regression_type == "Экспоненциальная":
            if (y_train <= 0).any() or (y_test <= 0).any():
                st.error("Экспоненциальная регрессия требует положительных значений в целевой переменной.")
                return
            y_train_log = np.log(y_train)
            y_test_log = np.log(y_test)
            model = LinearRegression()
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred = np.exp(y_pred_log)
            coefficients = model.coef_
            intercept = model.intercept_
            feature_names = X.columns

        elif regression_type == "Степенная":
            if (X_train.values <= 0).any() or (y_train <= 0).any():
                st.error("Степенная регрессия требует положительных значений в данных.")
                return
            X_train_log = np.log(X_train)
            X_test_log = np.log(X_test)
            y_train_log = np.log(y_train)
            y_test_log = np.log(y_test)
            model = LinearRegression()
            model.fit(X_train_log, y_train_log)
            y_pred_log = model.predict(X_test_log)
            y_pred = np.exp(y_pred_log)
            coefficients = model.coef_
            intercept = model.intercept_
            feature_names = X.columns

        elif regression_type == "Lasso":
            model = make_pipeline(StandardScaler(), Lasso(alpha=0.1))

        elif regression_type == "SVR (Метод опорных векторов)":
            model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

        elif regression_type == "Decision Tree (Решающее дерево)":
            model = DecisionTreeRegressor(random_state=42)

        elif regression_type == "Random Forest (Случайный лес)":
            model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

        elif regression_type == "Gradient Boosting (Градиентный бустинг)":
            model = GradientBoostingRegressor(random_state=42)

        elif regression_type == "Gaussian Processes (Гауссовские процессы)":
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, random_state=42)

        elif regression_type == "Neural Network (Нейронная сеть)":
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0, validation_split=0.2)
            y_pred = model.predict(X_test).flatten()
            fig = px.line(history.history, y=['loss', 'val_loss'], labels={"value": "Loss", "index": "Epoch"})
            st.plotly_chart(fig)

        if regression_type not in ["Экспоненциальная", "Степенная", "Neural Network (Нейронная сеть)"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        st.session_state.regression_results = {
            'model': model,
            'X_train': X_train,
            'y_train': y_train,
            'regression_type': regression_type,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        st.session_state.results_calculated = True

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Метрики модели")
        st.write(f"MSE (Среднеквадратичная ошибка): {mse:.2f}")
        st.write(f"RMSE (Корень из среднеквадратичной ошибки): {rmse:.2f}")
        st.write(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
        st.write(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")
        st.write(f"R² (Коэффициент детерминации): {r2:.2f}")

        if regression_type in ["Линейная", "Квадратическая", "Кубическая", "Логарифмическая", "Lasso",
                               "Экспоненциальная", "Степенная"]:
            if regression_type in ["Линейная", "Квадратическая", "Кубическая", "Логарифмическая", "Lasso"]:
                coefficients = model.named_steps["linearregression"].coef_ if regression_type != "Lasso" else \
                model.named_steps["lasso"].coef_
                intercept = model.named_steps["linearregression"].intercept_ if regression_type != "Lasso" else \
                model.named_steps["lasso"].intercept_
                feature_names = (
                    X.columns
                    if regression_type == "Линейная"
                    else model.named_steps["polynomialfeatures"].get_feature_names_out(X.columns)
                    if regression_type in ["Квадратическая", "Кубическая"]
                    else X.columns
                )
            show_formula(coefficients, intercept, feature_names, regression_type)
            if regression_type != "Экспоненциальная" and regression_type != "Степенная":
                show_feature_importance(coefficients, feature_names)

        st.subheader("Фактические vs предсказанные значения")
        fig1 = px.scatter(x=y_test, y=y_pred, labels={"x": "Фактические значения", "y": "Предсказанные значения"})
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines",
                                  name="Идеальная линия"))
        st.plotly_chart(fig1)

        st.subheader("График остатков")
        residuals = y_test - y_pred
        fig2 = px.scatter(x=y_pred, y=residuals, labels={"x": "Предсказанные значения", "y": "Остатки"})
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2)

    except ValueError as e:
        st.error(f"Ошибка: {e}")
    except Exception as e:
        st.error(f"Ошибка при выполнении регрессии: {e}")


def save_model_to_file(model, regression_type):
    if model is None:
        st.error("Модель не обучена. Сначала обучите модель.")
        return

    try:
        if regression_type == "Neural Network (Нейронная сеть)":
            file_format = st.selectbox("Выберите формат файла", [".h5"], key="nn_format")
        else:
            file_format = st.selectbox("Выберите формат файла", [".pkl", ".joblib"], key="ml_format")

        default_filename = f"model_{regression_type[:10]}{file_format}"
        file_path = st.text_input("Имя файла для сохранения", value=default_filename, key="save_path")

        if st.button("Сохранить модель", key="save_btn"):
            if regression_type == "Neural Network (Нейронная сеть)":
                save_model(model, file_path)
            else:
                if file_format == ".pkl":
                    joblib.dump(model, file_path)
                elif file_format == ".joblib":
                    joblib.dump(model, file_path)

            with open(file_path, "rb") as f:
                st.download_button(
                    label="Скачать модель",
                    data=f,
                    file_name=file_path,
                    mime="application/octet-stream",
                    key=f"dl_{file_path}"
                )
            st.success(f"Модель сохранена: {file_path}")
            os.remove(file_path)

    except Exception as e:
        st.error(f"Ошибка при сохранении: {e}")


# Основной интерфейс
st.title("Анализ регрессии и оптимизация")

df = load_data()
if df is not None:
    show_correlation_heatmaps(df)

    regression_types = [
        "Линейная", "Квадратическая", "Кубическая", "Логарифмическая",
        "Экспоненциальная", "Степенная", "Lasso",
        "SVR (Метод опорных векторов)", "Decision Tree (Решающее дерево)",
        "Random Forest (Случайный лес)", "Gradient Boosting (Градиентный бустинг)",
        "Gaussian Processes (Гауссовские процессы)", "Neural Network (Нейронная сеть)"
    ]
    regression_type = st.selectbox("Тип регрессии", regression_types, key="reg_type")

    if st.button("Выполнить регрессию", key="run_reg"):
        run_regression(df, regression_type)

    if st.session_state.results_calculated:
        st.subheader("Оптимизация параметров")
        results = st.session_state.regression_results

        opt_type = st.selectbox("Тип оптимизации", ["Минимизация", "Максимизация"], key="opt_type")

        bounds_df = pd.DataFrame({
            'Фактор': results['X_train'].columns,
            'Минимум': results['X_train'].min().values,
            'Максимум': results['X_train'].max().values
        })

        edited_bounds = st.data_editor(bounds_df, num_rows="fixed", key="bounds_edit")
        bounds = Bounds(edited_bounds['Минимум'].values, edited_bounds['Максимум'].values)

        if st.button("Запустить оптимизацию", key="run_opt"):
            run_optimization(
                results['model'],
                results['X_train'],
                results['y_train'],
                results['regression_type'],
                opt_type,
                bounds
            )

        if results['regression_type'] in [
            "SVR (Метод опорных векторов)", "Decision Tree (Решающее дерево)",
            "Random Forest (Случайный лес)", "Gradient Boosting (Градиентный бустинг)",
            "Gaussian Processes (Гауссовские процессы)", "Neural Network (Нейронная сеть)"
        ]:
            save_model_to_file(results['model'], results['regression_type'])
