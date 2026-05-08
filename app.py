import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title='Score de Crédito — IA',
    page_icon='💳',
    layout='wide'
)

load_dotenv()

CORES = {
    'BAIXO RISCO': '#2ecc71',
    'MÉDIO RISCO': '#f39c12',
    'ALTO RISCO' : '#e74c3c'
}

# ============================================================
# CARREGA MODELO E BANCO — mesmos artefatos do notebook
# ============================================================
@st.cache_resource
def carregar_modelo():
    modelo   = joblib.load('modelo_score_credito.pkl')
    scaler   = joblib.load('scaler_score_credito.pkl')
    features = joblib.load('features_score_credito.pkl')
    return modelo, scaler, features

@st.cache_resource
def conectar_banco():
    try:
        conn_str = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        engine = create_engine(conn_str)
        with engine.connect() as c:
            c.execute(text('SELECT 1'))
        return engine
    except Exception:
        return None

modelo, scaler, features = carregar_modelo()
engine = conectar_banco()

# ============================================================
# HEADER
# ============================================================
st.markdown("""<div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);
    padding:2rem;border-radius:12px;margin-bottom:1.5rem'>
    <h1 style='color:white;margin:0'>💳 Sistema Inteligente de Score de Crédito</h1>
    <p style='color:#a0aec0;margin:.5rem 0 0'>Análise preditiva de perfil financeiro com Machine Learning</p>
    </div>""", unsafe_allow_html=True)

aba1, aba2, aba3 = st.tabs(['🔍 Análise de Cliente', '📊 Dashboard Executivo', 'ℹ️ Sobre o Modelo'])

# ============================================================
# ABA 1 — PREDIÇÃO
# ============================================================
with aba1:
    st.subheader('📋 Dados do Cliente')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Dados Pessoais**')
        idade        = st.number_input('Idade', 18, 80, 35)
        estado_civil = st.selectbox('Estado Civil', [0,1,2],
                        format_func=lambda x: ['Solteiro','Casado','Divorciado'][x])
        qt_filhos    = st.number_input('Nº de Filhos', 0, 4, 1)
        escolaridade = st.selectbox('Escolaridade', [0,1,2,3],
                        format_func=lambda x: ['Fundamental','Médio','Superior','Pós-graduação'][x])
        uf           = st.number_input('UF (código numérico)', 0, 30, 5)

    with col2:
        st.markdown('**Situação Financeira**')
        ultimo_salario    = st.number_input('Último Salário (R$)', 0.0, 100000.0, 3500.0, step=500.0)
        outra_renda       = st.selectbox('Possui Outra Renda?', [0,1],
                                format_func=lambda x: 'Não' if x==0 else 'Sim')
        outra_renda_valor = st.number_input('Valor Outra Renda (R$)', 0.0, 50000.0, 0.0, step=500.0)
        trabalhando       = st.selectbox('Trabalhando Atualmente?', [0,1],
                                format_func=lambda x: 'Não' if x==0 else 'Sim')
        tempo_emprego     = st.number_input('Tempo no Emprego (meses)', 0, 480, 12)

    with col3:
        st.markdown('**Patrimônio**')
        casa_propria        = st.selectbox('Casa Própria?', [0,1],
                                    format_func=lambda x: 'Não' if x==0 else 'Sim')
        qt_imoveis          = st.number_input('Nº de Imóveis', 0, 10, 0)
        vl_imoveis          = st.number_input('Valor Total Imóveis (R$)', 0.0, 5000000.0, 0.0, step=10000.0)
        qt_carros           = st.number_input('Nº de Carros', 0, 10, 0)
        valor_tabela_carros = st.number_input('Valor Total Carros (R$)', 0.0, 500000.0, 0.0, step=5000.0)

    faixa_etaria = 0 if idade <= 30 else 1 if idade <= 40 else 2 if idade <= 50 else 3

    st.divider()

    # Botões de perfil de exemplo
    st.markdown('**⚡ Simular perfil de exemplo:**')
    c1e, c2e, c3e = st.columns(3)

    if c1e.button('👑 Cliente Premium', use_container_width=True):
        st.session_state.perfil_ex = 'premium'
        st.rerun()
    if c2e.button('👤 Cliente Médio', use_container_width=True):
        st.session_state.perfil_ex = 'medio'
        st.rerun()
    if c3e.button('⚠️ Alto Risco', use_container_width=True):
        st.session_state.perfil_ex = 'risco'
        st.rerun()

    analisar = st.button('🔍 Analisar Perfil', use_container_width=True, type='primary')

    if analisar:
        dados = {
            'UF': uf, 'IDADE': idade, 'ESCOLARIDADE': escolaridade,
            'ESTADO_CIVIL': estado_civil, 'QT_FILHOS': qt_filhos,
            'CASA_PROPRIA': casa_propria, 'QT_IMOVEIS': qt_imoveis,
            'VL_IMOVEIS': vl_imoveis, 'OUTRA_RENDA': outra_renda,
            'OUTRA_RENDA_VALOR': outra_renda_valor,
            'TEMPO_ULTIMO_EMPREGO_MESES': tempo_emprego,
            'TRABALHANDO_ATUALMENTE': trabalhando,
            'ULTIMO_SALARIO': ultimo_salario,
            'QT_CARROS': qt_carros,
            'VALOR_TABELA_CARROS': valor_tabela_carros,
            'FAIXA_ETARIA': faixa_etaria,
        }

        entrada = [dados[f] for f in features if f in dados]
        X = np.array(entrada).reshape(1, -1)
        X = scaler.transform(X)

        score = float(modelo.predict(X).flatten()[0])
        score = max(0, min(1000, score))

        if score >= 70:
            risco = 'BAIXO RISCO'; emoji = '✅'; cor = '#2ecc71'
            recomendacao = 'Cliente apto para crédito — baixa probabilidade de inadimplência.'
        elif score >= 35 :
            risco = 'MÉDIO RISCO'; emoji = '⚠️'; cor = '#f39c12'
            recomendacao = 'Análise adicional recomendada — considerar garantias ou limite reduzido.'
        else:
            risco = 'ALTO RISCO'; emoji = '❌'; cor = '#e74c3c'
            recomendacao = 'Crédito não recomendado — alta probabilidade de inadimplência.'

        st.divider()
        st.subheader('📊 Resultado da Análise')

        m1, m2, m3 = st.columns(3)
        m1.metric('Score Predito', f'{score:.0f}', 'de 1000 pontos')
        m2.metric('Classificação', f'{emoji} {risco}')
        m3.metric('Faixa Etária', ['Até 30','31 a 40','41 a 50','Maior 50'][faixa_etaria])

        st.progress(int(score) / 100)

        st.markdown(f"""<div style='background:{cor}22;border-left:4px solid {cor};
            padding:1rem;border-radius:8px;margin-top:1rem'>
            <strong style='color:{cor}'>{emoji} {risco}</strong><br>{recomendacao}
            </div>""", unsafe_allow_html=True)

        # Gauge bar
        fig, ax = plt.subplots(figsize=(7, 1.2))
        ax.barh([''], [100], color='#ecf0f1', height=0.5)
        ax.barh([''], [35],  color='#e74c3c', height=0.5)
        ax.barh([''], [max(0, min(400, score))], color='#e74c3c', height=0.5)
        if score > 35:
            ax.barh([''], [min(score, 70) - 35], left=35, color='#f39c12', height=0.5)
        if score > 70:
            ax.barh([''], [score - 70], left=70, color='#2ecc71', height=0.5)
        ax.axvline(35, color='white', lw=2)
        ax.axvline(70, color='white', lw=2)
        ax.axvline(score, color='black', lw=2, linestyle='--')
        ax.text(score, 0.35, f'  {score:.0f}', va='bottom', fontweight='bold', fontsize=11)
        ax.text(200, -0.45, 'Alto Risco', ha='center', fontsize=8, color='#e74c3c')
        ax.text(550, -0.45, 'Médio Risco', ha='center', fontsize=8, color='#f39c12')
        ax.text(850, -0.45, 'Baixo Risco', ha='center', fontsize=8, color='#2ecc71')
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel('Score de Crédito')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# ABA 2 — DASHBOARD EXECUTIVO (lê da camada GOLD)
# ============================================================
with aba2:
    st.subheader('📊 Dashboard Executivo — Painel da Diretoria')
    st.caption('Dados lidos diretamente da camada gold do PostgreSQL (Arquitetura Medallion)')

    if engine is None:
        st.warning('⚠️ Banco de dados não conectado. Verifique o arquivo .env e execute o notebook.')
    else:
        try:
            with engine.connect() as conn:
                kpis     = pd.read_sql(text('SELECT * FROM gold.kpis_diretoria'), conn)
                df_risco = pd.read_sql(text('SELECT * FROM gold.resumo_por_risco'), conn)
                df_uf    = pd.read_sql(text('SELECT * FROM gold.resumo_por_uf ORDER BY media_score DESC LIMIT 10'), conn)
                df_faixa = pd.read_sql(text('SELECT * FROM gold.resumo_por_faixa_etaria ORDER BY "FAIXA_ETARIA"'), conn)

            # KPIs
            st.markdown('#### 🎯 KPIs Principais')
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric('Total de Clientes',   f"{int(kpis['total_clientes_analisados'].iloc[0]):,}")
            k2.metric('Score Médio Geral',   f"{kpis['score_medio_geral'].iloc[0]:.0f}")
            k3.metric('Salário Médio',       f"R$ {kpis['salario_medio_geral'].iloc[0]:,.0f}")
            k4.metric('% Alto Risco',        f"{kpis['pct_alto_risco'].iloc[0]:.1f}%")
            k5.metric('Patrimônio Médio',    f"R$ {kpis['patrimonio_imovel_medio'].iloc[0]:,.0f}")

            st.divider()

            cg1, cg2 = st.columns(2)

            # Gráfico 1: Distribuição de risco
            with cg1:
                st.markdown('#### Clientes por Classificação de Risco')
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                cores_b = [CORES.get(r, '#95a5a6') for r in df_risco['classificacao']]
                bars = ax1.bar(df_risco['classificacao'], df_risco['total_clientes'],
                               color=cores_b, edgecolor='white', linewidth=1.5)
                for bar, pct in zip(bars, df_risco['percentual']):
                    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                             f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=10)
                ax1.set_ylabel('Nº de Clientes')
                ax1.set_title('Volume por Perfil de Risco')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()

            # Gráfico 2: Top UF por score
            with cg2:
                st.markdown('#### Score Médio por UF (Top 10)')
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.barh(df_uf['UF'].astype(str), df_uf['media_score'],
                         color='#3498db', edgecolor='white')
                ax2.set_xlabel('Score Médio')
                ax2.set_title('Ranking de UF por Score')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            cg3, cg4 = st.columns(2)

            # Gráfico 3: Score por faixa etária
            with cg3:
                st.markdown('#### Score Médio por Faixa Etária')
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                ax3.bar(df_faixa['FAIXA_ETARIA'].astype(str), df_faixa['media_score'],
                        color='#9b59b6', edgecolor='white')
                ax3.set_ylabel('Score Médio')
                ax3.set_title('Score por Faixa Etária')
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

            # Gráfico 4: Salário médio por risco
            with cg4:
                st.markdown('#### Salário Médio por Perfil de Risco')
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                cores_sal = [CORES.get(r, '#95a5a6') for r in df_risco['classificacao']]
                ax4.bar(df_risco['classificacao'], df_risco['media_salario'],
                        color=cores_sal, edgecolor='white')
                ax4.set_ylabel('Salário Médio (R$)')
                ax4.set_title('Salário Médio por Perfil')
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()

            st.divider()
            st.markdown('#### 📋 Tabela Completa — Resumo por Classificação de Risco')
            st.dataframe(
                df_risco.rename(columns={
                    'classificacao'     : 'Classificação',
                    'total_clientes'    : 'Total Clientes',
                    'percentual'        : '% Carteira',
                    'media_score'       : 'Score Médio',
                    'media_salario'     : 'Salário Médio (R$)',
                    'media_idade'       : 'Idade Média',
                    'media_filhos'      : 'Média Filhos',
                    'media_valor_imoveis': 'Patrimônio Médio (R$)'
                }),
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f'Erro ao carregar dados do banco: {e}')
            st.info('Execute todas as células do notebook para popular as camadas gold.')

# ============================================================
# ABA 3 — SOBRE O MODELO
# ============================================================
with aba3:
    st.subheader('ℹ️ Sobre o Modelo e a Arquitetura')
    ci1, ci2 = st.columns(2)

    with ci1:
        st.markdown("""#### 🤖 Modelo de Machine Learning
- **Algoritmo:** Regressão Linear
- **Variável alvo:** Score de Crédito (0–100)
- **Normalização:** MinMaxScaler
- **Split:** 70% treino / 30% teste
- **Serialização:** joblib (.pkl)

#### 📏 Critérios de Classificação de Risco
| Faixa de Score | Classificação |
|---|---|
| 70 – 100 | ✅ Baixo Risco |
| 35 – 69  | ⚠️ Médio Risco |
| 0 – 34    | ❌ Alto Risco  |
""")

    with ci2:
        st.markdown("""#### 🏗️ Arquitetura Medallion
```
dados_credito.csv
      ↓  ingestão
bronze.credito_raw
      ↓  ETL (limpeza + encoding)
silver.credito_tratado
      ↓  ML + agregações SQL
gold.predicoes_score
gold.resumo_por_uf
gold.resumo_por_risco
gold.resumo_por_faixa_etaria
gold.kpis_diretoria
      ↓
Power BI + Streamlit
```

#### 🔧 Stack Tecnológico
- **Python** — pandas, scikit-learn, matplotlib
- **PostgreSQL** — Docker (porta 5433)
- **SQLAlchemy** — ORM e queries SQL
- **Streamlit** — interface web de produção
""")

    st.divider()
    st.markdown('#### 📌 Features Utilizadas no Modelo')
    st.code(str(features), language='python')