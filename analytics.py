import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import japanize_matplotlib
import re

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'

# データの読み込み
@st.cache_data
def load_data():
    with open('talentdata.csv', 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    try:
        df = pd.read_csv('talentdata.csv', encoding=encoding)
        # ％を含む列を除外
        columns_to_drop = ['ドラマ関心度（シリアス）％', 'ドラマ関心度（コメディ）％', 'バラエティ関心度％', 
                           '映画関心度（シリアス）％', '映画関心度（コメディ）％', '雑誌関心度％']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        return df
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {str(e)}")
        return None

# スコアの正規化
def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

# 総合関心度の計算
def calculate_comprehensive_interest(row, genre_cols, weights):
    return sum(row[col] * weights.get(f'{col}_weight', 0.1) for col in genre_cols)

# 総合スコアの計算
def calculate_total_score(recognition, popularity, comprehensive_interest, weights, operators):
    recognition_weight = weights['recognition_weight']
    popularity_weight = weights['popularity_weight']
    interest_weight = weights['interest_weight']
    
    recognition_score = recognition * recognition_weight
    popularity_score = popularity * popularity_weight
    interest_score = comprehensive_interest * interest_weight
    
    total_score = recognition_score
    
    for score, op in zip([popularity_score, interest_score], operators):
        if op == '✕':
            total_score *= score
        elif op == '＋':
            total_score += score
    
    return total_score

# 補正係数の計算
def correction_factor(recognizability, respondents):
    return (1 / (recognizability / 100)) * (respondents.max() / respondents)

# 関心度の補正
def correct_interest_scores(df, interest_cols, recognition_col, respondents_col):
    for col in interest_cols:
        corrected_col = f"補正後{col}"
        df[corrected_col] = df[col] * correction_factor(df[recognition_col], df[respondents_col])
    return df

# 補正後の総合スコアの計算
def calculate_corrected_total_score(df, recognition_col, popularity_col, genre_cols, respondents_col, weights, operators):
    df['補正後人気度'] = df[popularity_col] * correction_factor(df[recognition_col], df[respondents_col])
    df = correct_interest_scores(df, genre_cols, recognition_col, respondents_col)
    df['補正後総合関心度'] = df.apply(lambda row: calculate_comprehensive_interest(row, [f'補正後{col}' for col in genre_cols], weights), axis=1)
    df['補正後総合スコア'] = df.apply(lambda row: calculate_total_score(row[recognition_col], row['補正後人気度'], row['補正後総合関心度'], weights, operators), axis=1)
    return df

# タレント名のカッコを外す関数
def clean_name(name):
    return re.sub(r'\(.*?\)', '', name).strip()

# メイン関数
def main():
    st.set_page_config(page_title="タレントデータ分析アプリ", layout="wide")
    st.title('タレントデータ分析アプリ')

    df = load_data()
    if df is None:
        return

    # タレント名のカッコを外す
    df['タレント名'] = df['タレント名'].apply(clean_name)

    # 列名の確認と設定
    name_col = "タレント名"
    recognition_col = "知名度"
    popularity_col = "人気度"
    respondents_col = "有効回答者数"
    all_genre_cols = [col for col in df.columns if '関心度' in col and not col.endswith('％')]

    st.write("使用する列名:")
    st.write(f"タレント名: {name_col}")
    st.write(f"知名度: {recognition_col}")
    st.write(f"人気度: {popularity_col}")
    st.write(f"ジャンル列: {all_genre_cols}")

    # 補正値の入力
    st.sidebar.header("重みの設定")
    recognition_weight = st.sidebar.slider("知名度の重み", 0.0, 1.0, 0.7, 0.05)
    popularity_weight = st.sidebar.slider("人気度の重み", 0.0, 1.0, 0.2, 0.05)
    interest_weight = st.sidebar.slider("総合関心度の重み", 0.0, 1.0, 0.1, 0.05)
    
    weights = {
        'recognition_weight': recognition_weight,
        'popularity_weight': popularity_weight,
        'interest_weight': interest_weight,
    }

    # 各関心度の重みを個別に設定
    st.sidebar.subheader("関心度の重み")
    for col in all_genre_cols:
        weight_key = f'{col}_weight'
        weights[weight_key] = st.sidebar.slider(f"{col}の重み", 0.0, 1.0, 0.1, 0.05)

    # 演算子の選択
    st.sidebar.subheader("演算子の選択")
    operators = []
    operators.append(st.sidebar.selectbox("知名度と人気度の演算子", ['✕', '＋'], key='op_popularity'))
    operators.append(st.sidebar.selectbox("前の結果と総合関心度の演算子", ['✕', '＋'], key='op_interest'))

    # 総合スコアの数式を表示
    st.sidebar.markdown("### 総合スコアの数式")
    formula = f"総合スコア = (知名度 * {recognition_weight}) {operators[0]} (人気度 * {popularity_weight}) {operators[1]} (総合関心度 * {interest_weight})"
    st.sidebar.markdown(formula)

    # 総合関心度の計算
    df['総合関心度'] = df.apply(lambda row: calculate_comprehensive_interest(row, all_genre_cols, weights), axis=1)

    # スコアの正規化
    df['知名度（正規化）'] = normalize_series(df[recognition_col])
    df['人気度（正規化）'] = normalize_series(df[popularity_col])
    df['総合関心度（正規化）'] = normalize_series(df['総合関心度'])
    for col in all_genre_cols:
        df[f'{col}（正規化）'] = normalize_series(df[col])

    # 総合スコアの計算
    df['総合スコア'] = df.apply(lambda row: calculate_total_score(row[recognition_col], row[popularity_col], row['総合関心度'], weights, operators), axis=1)

    # 補正後の総合スコアの計算
    df = calculate_corrected_total_score(df, recognition_col, popularity_col, all_genre_cols, respondents_col, weights, operators)

    # 補正後のスコアの正規化
    df['補正後人気度（正規化）'] = normalize_series(df['補正後人気度'])
    df['補正後総合関心度（正規化）'] = normalize_series(df['補正後総合関心度'])
    for col in all_genre_cols:
        df[f'補正後{col}（正規化）'] = normalize_series(df[f'補正後{col}'])
    df['補正後総合スコア（正規化）'] = normalize_series(df['補正後総合スコア'])

    # 値の種類を選択するオプションを追加
    value_type = st.sidebar.radio("表示する値の種類を選択してください", ["元の値", "正規化された値"])

    # サイドバー
    analysis_option = st.sidebar.selectbox(
        "分析項目を選択してください",
        ["タレント総合ランキング", "分布図", "相関分析", "タレント比較", "タレント検索・フィルター"]
    )

    if analysis_option == "タレント総合ランキング":
        st.header("タレント総合ランキング")

        # 並び替え機能
        sort_options = ['総合スコア', '補正後総合スコア', recognition_col, popularity_col, '総合関心度'] + all_genre_cols
        if value_type == "正規化された値":
            sort_options = ['総合スコア', '補正後総合スコア（正規化）', '知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
        
        sort_by = st.selectbox("並び替えの基準", sort_options)
        sort_order = st.radio("並び替え順序", ['昇順', '降順'], index=1)
        sort_ascending = True if sort_order == '昇順' else False

        sorted_df = df.sort_values(sort_by, ascending=sort_ascending)
        
        display_columns = [name_col, '総合スコア', '補正後総合スコア', recognition_col, popularity_col, '総合関心度'] + all_genre_cols
        if value_type == "正規化された値":
            display_columns = [name_col, '総合スコア', '補正後総合スコア（正規化）', '知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
        
        st.table(sorted_df[display_columns])

    elif analysis_option == "分布図":
        st.header("スコアの分布図")

        score_options = ['総合スコア', '補正後総合スコア', recognition_col, popularity_col, '総合関心度'] + all_genre_cols
        if value_type == "正規化された値":
            score_options = ['総合スコア', '補正後総合スコア（正規化）', '知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]

        selected_score = st.selectbox("表示するスコアを選択してください", score_options)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_score], kde=True, ax=ax)
        ax.set_title(f"{selected_score}の分布")
        ax.set_xlabel(selected_score)
        ax.set_ylabel("頻度")
        st.pyplot(fig)

    elif analysis_option == "相関分析":
        st.header("相関分析")
        
        correlation_columns = [recognition_col, popularity_col, '総合関心度'] + all_genre_cols
        if value_type == "正規化された値":
            correlation_columns = ['知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
        
        corr_matrix = df[correlation_columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("相関行列")
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_option == "タレント比較":
        st.header("タレント比較")
        talents = st.multiselect("比較するタレントを選択してください（最大5人）", df[name_col].tolist(), max_selections=5)
        
        if talents:
            comparison_df = df[df[name_col].isin(talents)]
            
            if comparison_df.empty:
                st.warning("選択されたタレントのデータが見つかりません。")
            else:
                if value_type == "元の値":
                    columns_to_display = [name_col, '総合スコア', '補正後総合スコア', recognition_col, popularity_col, '総合関心度'] + all_genre_cols
                    chart_columns = ['総合関心度'] + all_genre_cols
                else:
                    columns_to_display = [name_col, '総合スコア', '補正後総合スコア（正規化）', '知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
                    chart_columns = ['総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
                
                comparison_data = comparison_df[columns_to_display].set_index(name_col)
                st.table(comparison_data)

                st.subheader("レーダーチャート比較")
                
                # レーダーチャートの作成
                num_vars = len(chart_columns)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1]  # 最初の点を最後にも追加して閉じる

                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

                for talent in talents:
                    if talent in comparison_df[name_col].values:
                        talent_data = comparison_df[comparison_df[name_col] == talent].iloc[0]
                        values = [talent_data[col] for col in chart_columns]
                        values += values[:1]  # 最初の値を最後にも追加して閉じる
                        ax.plot(angles, values, 'o-', linewidth=2, label=talent)
                        ax.fill(angles, values, alpha=0.25)
                    else:
                        st.warning(f"{talent}のデータが見つかりません。")

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(chart_columns, size=8)
                
                if value_type == "正規化された値":
                    ax.set_ylim(0, 1)
                    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                else:
                    max_value = max([comparison_df[col].max() for col in chart_columns])
                    ax.set_ylim(0, max_value)
                    ax.set_yticks(np.linspace(0, max_value, 5))
                
                ax.set_title(f"タレント比較レーダーチャート（{value_type}）")

                # グリッド線の調整
                ax.set_rlabel_position(0)
                ax.grid(True)

                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                st.pyplot(fig)
        else:
            st.info("比較するタレントを選択してください。")

    elif analysis_option == "タレント検索・フィルター":
        st.header("タレント検索・フィルター")

        # 検索機能
        search_term = st.text_input("タレント名で検索", "")
        if search_term:
            search_results = df[df[name_col].str.contains(search_term, na=False, case=False)]
            display_columns = [name_col, '総合スコア', '補正後総合スコア', recognition_col, popularity_col, '総合関心度'] + all_genre_cols
            if value_type == "正規化された値":
                display_columns = [name_col, '総合スコア', '補正後総合スコア（正規化）', '知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
            st.table(search_results[display_columns])

        # 並び替え機能
        sort_options = ['総合スコア', '補正後総合スコア', recognition_col, popularity_col, '総合関心度'] + all_genre_cols
        if value_type == "正規化された値":
            sort_options = ['総合スコア', '補正後総合スコア（正規化）', '知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]
        
        sort_by = st.selectbox("並び替えの基準", sort_options)
        sort_order = st.radio("並び替え順序", ['昇順', '降順'])
        sort_ascending = True if sort_order == '昇順' else False
        sorted_df = df.sort_values(sort_by, ascending=sort_ascending)

        # フィルター機能
        st.subheader("フィルター")
        filter_columns = [recognition_col, popularity_col, '総合関心度'] + all_genre_cols
        if value_type == "正規化された値":
            filter_columns = ['知名度（正規化）', '人気度（正規化）', '総合関心度（正規化）'] + [f'{col}（正規化）' for col in all_genre_cols]

        filters = {}
        for col in filter_columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            filters[col] = st.slider(f"{col}の範囲", min_value=min_val, max_value=max_val, value=(min_val, max_val))

        filtered_df = sorted_df
        for col, (min_val, max_val) in filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        st.write(f"フィルター適用後のタレント数: {len(filtered_df)}")
        st.table(filtered_df[display_columns])

    st.sidebar.info("このアプリは、タレントデータの分析と可視化を行います。サイドバーから分析項目を選択してください。")

if __name__ == "__main__":
    main()