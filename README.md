# 💳 Sistema Inteligente de Análise de Perfil Financeiro

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-blue?logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em%20Produção-brightgreen)

> Solução de ponta a ponta para análise de perfil financeiro, consumo e geração de insights baseada em dados — integrando engenharia de dados, machine learning e visualização interativa.

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Demonstração](#-demonstração)
- [Arquitetura da Solução](#️-arquitetura-da-solução)
- [Stack Tecnológico](#-stack-tecnológico)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Como Executar](#-como-executar)
- [Modelo de Machine Learning](#-modelo-de-machine-learning)
- [Arquitetura Medallion](#️-arquitetura-medallion)
- [Dashboard Streamlit](#-dashboard-streamlit)
- [Critérios de Classificação de Risco](#-critérios-de-classificação-de-risco)
- [Equipe](#-equipe)

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como entregável final de um **Hackathon de Dados**, com o objetivo de construir uma solução completa de análise de perfil financeiro e geração de score de crédito utilizando Machine Learning.

### Objetivos

- ✅ Extração, simulação e limpeza de dados via arquivos CSV e banco de dados relacional
- ✅ Processamento de dados com Python e consultas SQL estruturadas
- ✅ Aplicação de estatística descritiva e análise exploratória
- ✅ Desenvolvimento de dashboard analítico interativo com Streamlit
- ✅ Desenvolvimento de modelo preditivo de Machine Learning
- ✅ Deploy do modelo em produção

---

## 🎬 Demonstração

🌐 **App em produção:** [score-credito.streamlit.app](https://score-credito.streamlit.app)

O sistema conta com **3 módulos principais**:

| Módulo | Descrição |
|--------|-----------|
| 🔍 **Análise de Cliente** | Predição do score de crédito em tempo real com classificação de risco |
| 📊 **Dashboard Executivo** | Painel da Diretoria com KPIs e gráficos lidos da camada Gold do PostgreSQL |
| ℹ️ **Sobre o Modelo** | Documentação da arquitetura Medallion, stack tecnológico e features do modelo |

---

## 🏗️ Arquitetura da Solução

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUXO DE DADOS                           │
│                                                             │
│  dados_credito.csv                                          │
│         │                                                   │
│         ▼  ingestão bruta                                   │
│  ┌─────────────────┐                                        │
│  │  bronze schema  │  ← dados originais preservados         │
│  └────────┬────────┘                                        │
│           │  ETL: limpeza + encoding + feature engineering  │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  silver schema  │  ← dados tratados e prontos            │
│  └────────┬────────┘                                        │
│           │  Modelo ML + agregações SQL                     │
│           ▼                                                 │
│  ┌─────────────────────────────────────┐                    │
│  │           gold schema               │                    │
│  │  • predicoes_score                  │                    │
│  │  • resumo_por_uf                    │                    │
│  │  • resumo_por_risco                 │                    │
│  │  • resumo_por_faixa_etaria          │                    │
│  │  • kpis_diretoria                   │                    │
│  └────────┬────────────────────────────┘                    │
│           │                                                 │
│     ┌─────┴──────┐                                          │
│     ▼            ▼                                          │
│  Power BI    Streamlit                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Stack Tecnológico

### Linguagem & Análise
| Tecnologia | Uso |
|---|---|
| Python 3.10+ | Linguagem principal |
| pandas | Manipulação de dados |
| numpy | Computação numérica |
| matplotlib | Visualizações |
| DuckDB | Queries exploratórias em memória |

### Banco de Dados
| Tecnologia | Uso |
|---|---|
| PostgreSQL 18 | Banco principal (Arquitetura Medallion) |
| Docker | Containerização do banco |
| SQLAlchemy | ORM e conexão Python-PostgreSQL |

### Machine Learning
| Tecnologia | Uso |
|---|---|
| scikit-learn | Treinamento e avaliação do modelo |
| joblib | Serialização dos artefatos (.pkl) |

### Produção
| Tecnologia | Uso |
|---|---|
| Streamlit | Interface web interativa |
| Streamlit Cloud | Deploy em nuvem gratuito |

---

## 📁 Estrutura do Projeto

```
score-credito/
│
├── 📓 ScoreCredito.ipynb          # Notebook principal — pipeline completo
│
├── 🌐 app.py                      # Interface Streamlit (produção)
│
├── 🤖 modelo_score_credito.pkl    # Modelo treinado serializado
├── ⚖️  scaler_score_credito.pkl   # MinMaxScaler serializado
├── 📋 features_score_credito.pkl  # Lista de features do modelo
│
├── 📦 requirements.txt            # Dependências do projeto
├── 🔒 .gitignore                  # Arquivos ignorados pelo git
├── 🔑 .env.example                # Template de variáveis de ambiente
│
└── 📁 .streamlit/
    └── config.toml                # Configurações do Streamlit
```

---

## ✅ Pré-requisitos

- Python 3.10+
- Docker Desktop instalado e rodando
- Git

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/Elbiabuglio/score-credito.git
cd score-credito
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite com suas credenciais
```

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=data_projects_dw
DB_USER=postgres
DB_PASSWORD=sua_senha
```

### 4. Suba o PostgreSQL com Docker

```bash
docker run --name postgres \
  -e POSTGRES_PASSWORD=1234 \
  -p 5433:5432 \
  -d postgres
```

### 5. Execute o notebook completo

Abra o `ScoreCredito.ipynb` no VS Code ou Jupyter e execute todas as células em ordem. O notebook irá:

- Carregar e limpar os dados
- Popular as camadas Bronze, Silver e Gold no PostgreSQL
- Treinar e salvar o modelo
- Gerar o `app.py` do Streamlit

### 6. Inicie o Streamlit

```bash
streamlit run app.py
```

Acesse em: **http://localhost:8501**

---

## 🤖 Modelo de Machine Learning

### Algoritmo
**Regressão Linear** com normalização via `MinMaxScaler`

### Pipeline
```
Dados Brutos → LabelEncoder → MinMaxScaler → LinearRegression → Score (0–100)
```

### Split
| Conjunto | Proporção |
|---|---|
| Treino | 70% |
| Teste | 30% |

### Métricas de Avaliação
| Métrica | Descrição |
|---|---|
| R² | Coeficiente de determinação |
| MAE | Erro médio absoluto |
| RMSE | Raiz do erro quadrático médio |

### Features Utilizadas

```python
['UF', 'IDADE', 'ESCOLARIDADE', 'ESTADO_CIVIL', 'QT_FILHOS',
 'CASA_PROPRIA', 'QT_IMOVEIS', 'VL_IMOVEIS', 'OUTRA_RENDA',
 'OUTRA_RENDA_VALOR', 'TEMPO_ULTIMO_EMPREGO_MESES',
 'TRABALHANDO_ATUALMENTE', 'ULTIMO_SALARIO',
 'QT_CARROS', 'VALOR_TABELA_CARROS', 'FAIXA_ETARIA']
```

---

## 🏅 Arquitetura Medallion

O projeto implementa a **Arquitetura Medallion** no PostgreSQL, organizando os dados em três camadas de qualidade crescente:

### 🥉 Bronze — Dados Brutos
```sql
schema: bronze
tabela: credito_raw
```
Armazena o CSV original sem qualquer tratamento. Garante rastreabilidade e auditoria dos dados de origem.

### 🥈 Silver — Dados Tratados
```sql
schema: silver
tabela: credito_tratado
```
Dados após o processo completo de ETL:
- Tratamento de valores nulos (`ULTIMO_SALARIO` → mediana)
- Remoção de outliers (`QT_FILHOS > 4`)
- Encoding de variáveis categóricas (`LabelEncoder`)
- Engenharia de atributos (`FAIXA_ETARIA`, `CLASSIFICACAO_RISCO`)

### 🥇 Gold — Dados Agregados
```sql
schema: gold
tabelas: predicoes_score | resumo_por_uf | resumo_por_risco
         resumo_por_faixa_etaria | kpis_diretoria
```
Tabelas desnormalizadas e pré-calculadas, prontas para consumo direto no Power BI e Streamlit.

---

## 📊 Dashboard Streamlit

### Aba 1 — Análise de Cliente
- Formulário com 15 variáveis do perfil financeiro
- Predição do score em tempo real
- Gauge visual de classificação de risco
- Barra de progresso proporcional ao score

### Aba 2 — Dashboard Executivo (Painel da Diretoria)
- **5 KPIs** principais: total de clientes, score médio, salário médio, % alto risco, patrimônio médio
- **4 gráficos**: distribuição de risco, ranking por UF, score por faixa etária, salário por perfil
- Tabela detalhada com resumo por classificação
- Dados lidos diretamente da camada Gold do PostgreSQL

### Aba 3 — Sobre o Modelo
- Documentação do algoritmo e métricas
- Diagrama da Arquitetura Medallion
- Stack tecnológico completo
- Lista de features do modelo

---

## 🎯 Critérios de Classificação de Risco

| Faixa de Score | Classificação | Recomendação |
|---|---|---|
| 70 – 100 | ✅ Baixo Risco | Cliente apto para crédito |
| 35 – 69 | ⚠️ Médio Risco | Análise adicional recomendada |
| 0 – 34 | ❌ Alto Risco | Crédito não recomendado |

> Os critérios foram calibrados com base nos percentis reais da distribuição do score na base de dados (escala 0–100).

---

## 🔐 Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto baseado no `.env.example`:

| Variável | Descrição |
|---|---|
| `DB_HOST` | Host do PostgreSQL |
| `DB_PORT` | Porta do banco (padrão: 5433) |
| `DB_NAME` | Nome do banco de dados |
| `DB_USER` | Usuário do banco |
| `DB_PASSWORD` | Senha do banco |

> ⚠️ **Nunca commite o arquivo `.env`** — ele está no `.gitignore`.

---

## 👩‍💻 Equipe

Desenvolvido por **Grupo 25 - Dados** como projeto de conclusão de hackathon de dados.

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<div align="center">
  <sub>Desenvolvido com 💙 como projeto de Hackathon de Dados Elas+ Tech</sub>
</div>
