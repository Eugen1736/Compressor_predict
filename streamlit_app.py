import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import IsolationForest

# --- helper functions -----------------------------------------------------
@st.cache_data(show_spinner=False)
def load_training_data(path: str = "brg_data.csv") -> pd.DataFrame:
    """Загрузить обучающий датасет."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def build_model(train_df: pd.DataFrame):
    """Построить модель на основе обучающего датасета и вернуть порог."""
    sensor_columns = {
        'Fтк4': ['Гармоника вращ. тел кач. подшипника передней опоры КВД', 'мм/с'],
        'Fc4': ['Гармоника вращ. сепаратора подшипника передней опоры КВД', 'мм/с'],
        'Fc2': ['Гарм. вращ. сепар. подш. задней опоры КНД', 'мм/с'],
        'Fc3': ['Гарм. вращ. сепар. подш. передней опоры КНД', 'мм/с'],
    }

    X_train = train_df[list(sensor_columns.keys())]
    model = IsolationForest(n_estimators=400, contamination=0.05, random_state=42)
    model.fit(X_train)

    # порог для раннего предупреждения берём 10-й перцентиль скор-а на train
    threshold = pd.Series(model.decision_function(X_train)).quantile(0.1)

    return model, threshold, sensor_columns


def preprocess_test(df: pd.DataFrame) -> pd.DataFrame:
    """Сделать индексами столбец Date&time и добавить производные фичи (как в ноутбуке)."""
    df = df.copy()
    if 'Date&time' in df.columns:
        df['Date&time'] = pd.to_datetime(df['Date&time'])
        df = df.set_index('Date&time')
    else:
        # если индекс уже установлен, просто убедимся, что он datetime
        df.index = pd.to_datetime(df.index)

    # добавить дробные производные для выбранных сенсоров
    dt = df.index.to_series().diff().dt.total_seconds()
    for i in ['Fтк4', 'Fc4', 'Fc2', 'Fc3']:
        if i in df.columns:
            df[f'd{i}'] = df[i].diff() / dt
    return df


def analyze(df: pd.DataFrame, model: IsolationForest, threshold: float, sensor_columns: dict):
    """Добавить метрики аномалий и вернуть дату первой подтверждённой аномалии."""
    df = df.copy()
    features = list(sensor_columns.keys())
    X = df[features]

    df['anomaly_score'] = model.decision_function(X)
    df['anomaly_flag'] = (model.predict(X) == -1).astype(int)

    window = 10
    df['anomaly_rate'] = df['anomaly_flag'].rolling(window=window).mean()
    df['anomaly_score_smooth'] = df['anomaly_score'].rolling(window=window).mean()
    df['smoothed_score_ewma'] = df['anomaly_score'].ewm(alpha=0.2).mean()

    df['early_warning'] = (df['anomaly_score_smooth'] < threshold).astype(int)
    N = 3
    df['confirmed_anomaly'] = (df['early_warning'].rolling(window=N).sum() == N).astype(int)

    first_true_anomaly = df['confirmed_anomaly'].idxmax()
    if first_true_anomaly is not pd.NaT and df.loc[first_true_anomaly, 'confirmed_anomaly'] == 1:
        return first_true_anomaly, df
    else:
        return None, df


# --- Streamlit UI ---------------------------------------------------------
# конфиг страницы (иконка и заголовок окна)
st.set_page_config(page_title="Анализ аномалий", page_icon="🐱")

# экран загрузки с заголовком и картинкой кота-рабочего
st.title("Анализ аномалий в данных подшипников турбины")
    
st.markdown(
    " В качестве тестовых доступны файлы `CS_1.csv` и `CS_2.csv`, а также любой загружаемый пользователем CSV."
)

# подготовка модели
train_df = load_training_data()
model, threshold, sensor_columns = build_model(train_df)

st.sidebar.header("Входные данные")
choice = st.sidebar.radio("Выберите тестовый датасет:",
                            options=["CS_1.csv", "CS_2.csv", "Upload your own"])

test_df = None
if choice in ["CS_1.csv", "CS_2.csv"]:
    test_df = pd.read_csv(choice)
elif choice == "Upload your own":
    uploaded = st.sidebar.file_uploader("Загрузите CSV файл", type=["csv"])
    if uploaded is not None:
        test_df = pd.read_csv(uploaded)

if test_df is None:
    st.info("Пожалуйста, в sidebar выберите или загрузите тестовый файл.")
    st.stop()

# предобработка и анализ
try:
    test_df = preprocess_test(test_df)
except Exception as e:
    st.error(f"Не удалось обработать тестовый датасет: {e}")
    st.stop()

first_anom, result_df = analyze(test_df, model, threshold, sensor_columns)

st.subheader("Результаты")
if first_anom is not None:
    st.success(f"Первая подтверждённая аномалия: **{first_anom}**")
else:
    st.warning("Стабильных аномалий не обнаружено.")

st.write("Порог для раннего предупреждения (10-й перцентиль на train):", threshold)

# график
plot_df = result_df[['anomaly_score', 'anomaly_score_smooth']].copy()
# добавим столбец с порогом, чтобы отобразить на графике
plot_df['threshold'] = threshold

# строим Altair график, чтобы задать цвета
plot_df = plot_df.reset_index().melt(id_vars='Date&time', value_vars=['anomaly_score','anomaly_score_smooth','threshold'],
                                      var_name='metric', value_name='value')

chart = alt.Chart(plot_df).mark_line().encode(
    x='Date&time:T',
    y='value:Q',
    color=alt.Color('metric:N', scale=alt.Scale(domain=['anomaly_score','anomaly_score_smooth','threshold'],
                                              range=['blue','red','yellow'])),
)
st.altair_chart(chart, use_container_width=True)

# показать таблицу
with st.expander("Показать подробные данные"):  # можно прокручивать
    st.dataframe(result_df.head(200))

# возможность скачать результат
csv = result_df.to_csv().encode('utf-8')
st.download_button("Скачать результаты как CSV", csv, "anomaly_results.csv", "text/csv")
