import streamlit as st
import pandas as pd
import sqlite3
import mysql.connector
import psycopg2
import pymongo
from sqlalchemy import create_engine, text, inspect
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import os
import json
import io
from typing import Dict, Any, Optional
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #f0f2f6;
    border-left: 5px solid #1f77b4;
    padding: 1rem;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Handles multiple database types and operations"""
    
    def __init__(self):
        self.connection = None
        self.engine = None
        self.db_type = None
    
    def connect_sqlite(self, db_path: str):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(db_path)
            self.engine = create_engine(f"sqlite:///{db_path}")
            self.db_type = "sqlite"
            return True, "Connected to SQLite successfully!"
        except Exception as e:
            return False, f"SQLite connection error: {str(e)}"
    
    def connect_mysql(self, host: str, user: str, password: str, database: str, port: int = 3306):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=host, user=user, password=password, 
                database=database, port=port
            )
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            self.db_type = "mysql"
            return True, "Connected to MySQL successfully!"
        except Exception as e:
            return False, f"MySQL connection error: {str(e)}"
    
    def connect_postgresql(self, host: str, user: str, password: str, database: str, port: int = 5432):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=host, user=user, password=password,
                database=database, port=port
            )
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            self.db_type = "postgresql"
            return True, "Connected to PostgreSQL successfully!"
        except Exception as e:
            return False, f"PostgreSQL connection error: {str(e)}"
    
    def create_sample_data(self):
        """Create sample tables with data for demonstration"""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # Create employees table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    salary REAL NOT NULL,
                    hire_date DATE NOT NULL,
                    age INTEGER NOT NULL
                )
            ''')
            
            # Create sales table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sales (
                    id INTEGER PRIMARY KEY,
                    employee_id INTEGER,
                    product TEXT NOT NULL,
                    amount REAL NOT NULL,
                    sale_date DATE NOT NULL,
                    region TEXT NOT NULL,
                    FOREIGN KEY (employee_id) REFERENCES employees (id)
                )
            ''')
            
            # Insert sample data
            employees_data = [
                (1, 'John Smith', 'Engineering', 75000, '2022-01-15', 28),
                (2, 'Sarah Johnson', 'Marketing', 65000, '2021-03-20', 32),
                (3, 'Mike Brown', 'Sales', 60000, '2023-05-10', 29),
                (4, 'Emily Davis', 'Engineering', 80000, '2020-11-05', 35),
                (5, 'David Wilson', 'HR', 55000, '2022-08-12', 31)
            ]
            
            sales_data = [
                (1, 1, 'Software License', 5000, '2024-01-15', 'North'),
                (2, 2, 'Marketing Campaign', 15000, '2024-01-20', 'South'),
                (3, 3, 'Product Sale', 8000, '2024-02-01', 'East'),
                (4, 1, 'Consulting Service', 12000, '2024-02-15', 'West'),
                (5, 4, 'Software License', 7000, '2024-03-01', 'North')
            ]
            
            cursor.executemany('INSERT OR IGNORE INTO employees VALUES (?, ?, ?, ?, ?, ?)', employees_data)
            cursor.executemany('INSERT OR IGNORE INTO sales VALUES (?, ?, ?, ?, ?, ?)', sales_data)
            
            self.connection.commit()
            return True, "Sample data created successfully!"
        
        return False, "Sample data creation only supported for SQLite currently"
    
    def get_table_info(self):
        """Get information about tables in the database"""
        if not self.engine:
            return []
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            table_info = []
            
            for table in tables:
                columns = inspector.get_columns(table)
                table_info.append({
                    'table': table,
                    'columns': [(col['name'], str(col['type'])) for col in columns]
                })
            
            return table_info
        except Exception as e:
            st.error(f"Error getting table info: {str(e)}")
            return []
    
    # def execute_query(self, query: str):
    #     """Execute SQL query and return results"""
    #     try:
    #         query = query.strip()
            
    #         # Determine if it's a SELECT query or a modification query
    #         query_type = query.upper().split()[0] if query else ""
            
    #         if query_type == "SELECT":
    #             # For SELECT queries, return data
    #             df = pd.read_sql_query(query, self.engine)
    #             return True, df
    #         else:
    #             # For INSERT, UPDATE, DELETE, CREATE, etc.
    #             with self.engine.connect() as conn:
    #                 result = conn.execute(text(query))
    #                 conn.commit()
                    
    #                 if query_type in ["INSERT", "UPDATE", "DELETE"]:
    #                     return True, f"Query executed successfully. Rows affected: {result.rowcount}"
    #                 else:
    #                     return True, "Query executed successfully."
                        
    #     except Exception as e:
    #         return False, str(e)
    
    def execute_query(self, query: str):
        """Execute SQL query and return results"""
        try:
            query = query.strip()
            if not query:
                return False, "Empty query provided"
            
            # Determine if it's a SELECT query or a modification query
            query_type = query.upper().split()[0]
            
            if query_type == "SELECT":
                # For SELECT queries, return data
                try:
                    df = pd.read_sql_query(query, self.engine)
                    return True, df
                except Exception as e:
                    return False, f"SELECT query error: {str(e)}"
            else:
                # For INSERT, UPDATE, DELETE, CREATE, etc.
                try:
                    with self.engine.connect() as conn:
                        with conn.begin():
                            result = conn.execute(text(query))
                            
                            if query_type in ["INSERT", "UPDATE", "DELETE"]:
                                affected_rows = getattr(result, 'rowcount', 0)
                                return True, f"Query executed successfully. Rows affected: {affected_rows}"
                            elif query_type in ["CREATE", "DROP", "ALTER"]:
                                return True, f"{query_type} operation completed successfully."
                            else:
                                return True, "Query executed successfully."
                except Exception as e:
                    return False, f"{query_type} query error: {str(e)}"
                            
        except Exception as e:
            return False, f"Query execution error: {str(e)}"

class AIQueryGenerator:
    """Handles AI-powered SQL query generation using Groq and LangChain"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = None
        self.sql_agent = None
        
    def initialize_llm(self, model_name: str = "llama3-70b-8192"):
        """Initialize Groq LLM"""
        try:
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model_name=model_name,
                temperature=0.1
            )
            return True, "LLM initialized successfully!"
        except Exception as e:
            return False, f"LLM initialization error: {str(e)}"
    
    def setup_sql_agent(self, db_engine):
        """Setup SQL agent with database"""
        try:
            db = SQLDatabase(db_engine)
            
            self.sql_agent = create_sql_agent(
                llm=self.llm,
                db=db,
                verbose=True,
                handle_parsing_errors=True
            )
            return True, "SQL Agent setup successfully!"
        except Exception as e:
            return False, f"SQL Agent setup error: {str(e)}"
    
    def generate_sql_query(self, natural_language_query: str, table_info: list):
        """Generate SQL query from natural language using LLM"""
        try:
            # Create context about available tables
            context = "Available tables and columns:\n"
            for table in table_info:
                context += f"Table: {table['table']}\n"
                for col_name, col_type in table['columns']:
                    context += f"  - {col_name} ({col_type})\n"
                context += "\n"
            
            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""
                Given the following database schema:
                {context}
                
                Convert this natural language query to SQL:
                {query}
                
                Rules:
                1. Return ONLY the SQL query, no explanations or additional text
                2. Do not include markdown formatting, backticks, or code blocks
                3. Only use tables and columns that exist in the schema
                4. Support all SQL operations: SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER
                5. Write clean, efficient SQL
                6. Use appropriate JOINs when needed
                7. Make sure the query is syntactically correct
                8. For CREATE TABLE, use appropriate data types for the database
                9. For INSERT, include all required columns
                10. For UPDATE/DELETE, always include WHERE clause for safety
                
                SQL Query (raw SQL only):
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.invoke({"context": context, "query": natural_language_query})
            
            # Clean up the result - extract SQL from various formats
            sql_query = result["text"] if isinstance(result, dict) and "text" in result else str(result)
            sql_query = self._extract_clean_sql(sql_query)
            
            return True, sql_query.strip()
        except Exception as e:
            return False, f"Query generation error: {str(e)}"
    
    def _extract_clean_sql(self, text: str) -> str:
        """Extract clean SQL from LLM response"""
        # Remove common prefixes and explanations
        lines = text.strip().split('\n')
        sql_lines = []
        
        in_sql_block = False
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and common explanatory phrases
            if not line:
                continue
            if line.lower().startswith(('here is', 'here\'s', 'the sql', 'sql query:', 'query:', 'answer:')):
                continue
            if line.lower().startswith(('this is', 'this query', 'explanation:', 'note:')):
                continue
            
            # Handle code blocks
            if line.startswith('```sql'):
                in_sql_block = True
                continue
            elif line.startswith('```'):
                if in_sql_block:
                    break
                continue
            
            # Check if line looks like SQL
            if (line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH')) or
                in_sql_block or
                (sql_lines and not line.lower().startswith(('the ', 'this ', 'note', 'explanation')))):
                sql_lines.append(line)
                in_sql_block = True
        
        # If no SQL found, return the first non-empty line
        if not sql_lines:
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith(('here is', 'the sql', 'this is')):
                    return line
        
        return ' '.join(sql_lines)
    
    def query_with_agent(self, question: str):
        """Use SQL agent to answer questions"""
        if not self.sql_agent:
            return False, "SQL Agent not initialized"
        
        try:
            result = self.sql_agent.invoke({"input": question})
            return True, result.get("output", result)
        except Exception as e:
            return False, f"Agent query error: {str(e)}"

class DataVisualizer:
    """Creates visualizations from query results"""
    
    @staticmethod
    def auto_visualize(df: pd.DataFrame, chart_type: str = "auto"):
        """Automatically create visualizations based on data"""
        if df.empty:
            return None
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if chart_type == "auto":
            if len(numeric_cols) >= 2:
                chart_type = "scatter"
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                chart_type = "bar"
            elif len(categorical_cols) >= 1:
                chart_type = "pie"
            else:
                return None
        
        try:
            if chart_type == "bar" and len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif chart_type == "line" and len(numeric_cols) >= 1:
                fig = px.line(df, x=df.columns[0], y=numeric_cols[0])
            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif chart_type == "pie" and len(categorical_cols) >= 1:
                if len(numeric_cols) >= 1:
                    fig = px.pie(df, names=categorical_cols[0], values=numeric_cols[0])
                else:
                    value_counts = df[categorical_cols[0]].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index)
            else:
                return None
            
            fig.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
            return fig
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            return None

def main():
    st.markdown('<h1 class="main-header">🤖 AI Data Analyst</h1>', unsafe_allow_html=True)
    st.markdown("Transform natural language questions into SQL queries and get instant insights!")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'ai_generator' not in st.session_state:
        st.session_state.ai_generator = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        groq_api_key = st.text_input("Groq API Key", type="password", 
                                   help="Enter your Groq API key to enable AI features")
        
        if groq_api_key and not st.session_state.ai_generator:
            st.session_state.ai_generator = AIQueryGenerator(groq_api_key)
            success, msg = st.session_state.ai_generator.initialize_llm()
            if success:
                st.success(msg)
            else:
                st.error(msg)
        
        st.header("🗄️ Database Connection")
        
        # Database type selection
        db_type = st.selectbox("Select Database Type", 
                             ["SQLite", "MySQL", "PostgreSQL"])
        
        if db_type == "SQLite":
            st.subheader("SQLite Configuration")
            db_option = st.radio("Choose option:", 
                                ["Create sample database", "Upload existing database"])
            
            if db_option == "Create sample database":
                if st.button("Create Sample Database"):
                    success, msg = st.session_state.db_manager.connect_sqlite("sample_database.db")
                    if success:
                        st.session_state.db_manager.create_sample_data()
                        st.session_state.connected = True
                        st.success("Sample database created with employee and sales data!")
                    else:
                        st.error(msg)
            
            else:
                uploaded_file = st.file_uploader("Choose SQLite database file", 
                                               type=['db', 'sqlite', 'sqlite3'])
                if uploaded_file:
                    with open("uploaded_database.db", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    success, msg = st.session_state.db_manager.connect_sqlite("uploaded_database.db")
                    if success:
                        st.session_state.connected = True
                        st.success(msg)
                    else:
                        st.error(msg)
        
        elif db_type == "MySQL":
            st.subheader("MySQL Configuration")
            mysql_host = st.text_input("Host", value="localhost")
            mysql_port = st.number_input("Port", value=3306)
            mysql_user = st.text_input("Username")
            mysql_password = st.text_input("Password", type="password")
            mysql_database = st.text_input("Database Name")
            
            if st.button("Connect to MySQL"):
                success, msg = st.session_state.db_manager.connect_mysql(
                    mysql_host, mysql_user, mysql_password, mysql_database, mysql_port
                )
                if success:
                    st.session_state.connected = True
                    st.success(msg)
                else:
                    st.error(msg)
        
        elif db_type == "PostgreSQL":
            st.subheader("PostgreSQL Configuration")
            pg_host = st.text_input("Host", value="localhost")
            pg_port = st.number_input("Port", value=5432)
            pg_user = st.text_input("Username")
            pg_password = st.text_input("Password", type="password")
            pg_database = st.text_input("Database Name")
            
            if st.button("Connect to PostgreSQL"):
                success, msg = st.session_state.db_manager.connect_postgresql(
                    pg_host, pg_user, pg_password, pg_database, pg_port
                )
                if success:
                    st.session_state.connected = True
                    st.success(msg)
                else:
                    st.error(msg)
    
    # Main content area
    if st.session_state.connected:
        # Setup SQL agent if AI is available
        if st.session_state.ai_generator and st.session_state.ai_generator.llm:
            if not st.session_state.ai_generator.sql_agent:
                success, msg = st.session_state.ai_generator.setup_sql_agent(
                    st.session_state.db_manager.engine
                )
                if not success:
                    st.warning(f"SQL Agent setup failed: {msg}")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI Query", "📊 Database Explorer", 
                                               "📝 Manual SQL", "📈 Visualizations", "🛠️ Database Operations"])
        
        with tab1:
            st.markdown('<h2 class="sub-header">Ask Questions in Natural Language</h2>', 
                       unsafe_allow_html=True)
            
            if not groq_api_key:
                st.warning("Please enter your Groq API key in the sidebar to use AI features.")
            else:
                # Natural language query input
                user_question = st.text_area(
                    "Ask a question about your data:",
                    placeholder="e.g., 'Show me the top 5 employees by salary' or 'What are the total sales by region?'",
                    height=100
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("🤖 Generate SQL Query", type="primary"):
                        if user_question:
                            table_info = st.session_state.db_manager.get_table_info()
                            
                            with st.spinner("Generating SQL query..."):
                                success, result = st.session_state.ai_generator.generate_sql_query(
                                    user_question, table_info
                                )
                                
                                if success:
                                    st.code(result, language="sql")
                                    st.session_state.generated_query = result
                                    
                                    # Execute the generated query
                                    success_exec, result = st.session_state.db_manager.execute_query(result)
                                    if success_exec:
                                        if isinstance(result, pd.DataFrame):
                                            st.success("Query executed successfully!")
                                            st.dataframe(result, use_container_width=True)
                                            
                                            # Auto-generate visualization for SELECT queries
                                            if not result.empty:
                                                fig = DataVisualizer.auto_visualize(result)
                                                if fig:
                                                    st.plotly_chart(fig, use_container_width=True)
                                            
                                            st.session_state.last_result = result
                                        else:
                                            # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                                            st.success(result)
                                    else:
                                        st.error(f"Query execution failed: {result}")
                                else:
                                    st.error(f"Query generation failed: {result}")
                
                with col2:
                    if st.button("🎯 Use AI Agent"):
                        if user_question and st.session_state.ai_generator.sql_agent:
                            with st.spinner("AI Agent processing your question..."):
                                success, result = st.session_state.ai_generator.query_with_agent(user_question)
                                
                                if success:
                                    st.success("AI Agent Response:")
                                    st.write(result)
                                else:
                                    st.error(result)
                        else:
                            st.error("AI Agent not available or no question provided.")
        
        with tab2:
            st.markdown('<h2 class="sub-header">Database Schema Explorer</h2>', 
                       unsafe_allow_html=True)
            
            table_info = st.session_state.db_manager.get_table_info()
            
            if table_info:
                for table in table_info:
                    with st.expander(f"📋 Table: {table['table']}"):
                        col_df = pd.DataFrame(table['columns'], columns=['Column', 'Data Type'])
                        st.dataframe(col_df, use_container_width=True)
                        
                        # Show sample data
                        if st.button(f"Show sample data from {table['table']}", 
                                   key=f"sample_{table['table']}"):
                            success, df = st.session_state.db_manager.execute_query(
                                f"SELECT * FROM {table['table']} LIMIT 5"
                            )
                            if success:
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.error(f"Error fetching sample data: {df}")
            else:
                st.info("No tables found in the database.")
        
        with tab3:
            st.markdown('<h2 class="sub-header">Manual SQL Query</h2>', 
                       unsafe_allow_html=True)
            
            # Manual SQL input
            manual_query = st.text_area("Enter your SQL query:", height=150,
                                      placeholder="SELECT * FROM employees WHERE salary > 70000;")
            
            if st.button("Execute SQL Query"):
                if manual_query.strip():
                    success, result = st.session_state.db_manager.execute_query(manual_query)
                    if success:
                        if isinstance(result, pd.DataFrame):
                            st.success("Query executed successfully!")
                            st.dataframe(result, use_container_width=True)
                            
                            # Download option for SELECT queries
                            csv = result.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                            
                            st.session_state.last_result = result
                        else:
                            # For non-SELECT queries
                            st.success(result)
                    else:
                        st.error(f"Query execution failed: {result}")
        
        with tab4:
            st.markdown('<h2 class="sub-header">Data Visualizations</h2>', 
                       unsafe_allow_html=True)
            
            if 'last_result' in st.session_state and not st.session_state.last_result.empty:
                df = st.session_state.last_result
                
                chart_type = st.selectbox("Choose visualization type:", 
                                        ["auto", "bar", "line", "scatter", "pie"])
                
                fig = DataVisualizer.auto_visualize(df, chart_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Unable to create visualization with the current data and chart type.")
                
                # Additional chart options
                st.subheader("Custom Visualizations")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox("X-axis:", categorical_cols + numeric_cols)
                    with col2:
                        y_axis = st.selectbox("Y-axis:", numeric_cols)
                    
                    if st.button("Create Custom Chart"):
                        fig = px.bar(df, x=x_axis, y=y_axis)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Execute a query first to see visualizations here.")
        
        with tab5:
            st.markdown('<h2 class="sub-header">Database Operations (CRUD)</h2>', 
                       unsafe_allow_html=True)
            
            operation = st.selectbox("Select Operation:", 
                                   ["Create Table", "Insert Data", "Update Data", "Delete Data", "Drop Table"])
            
            if operation == "Create Table":
                st.subheader("Create New Table")
                
                table_name = st.text_input("Table Name:")
                num_columns = st.number_input("Number of Columns:", min_value=1, max_value=20, value=3)
                
                columns = []
                col1, col2, col3 = st.columns(3)
                
                for i in range(num_columns):
                    with col1:
                        col_name = st.text_input(f"Column {i+1} Name:", key=f"col_name_{i}")
                    with col2:
                        col_type = st.selectbox(f"Column {i+1} Type:", 
                                              ["VARCHAR(255)", "INT", "FLOAT", "DATE", "TEXT", "BOOLEAN"],
                                              key=f"col_type_{i}")
                    with col3:
                        is_primary = st.checkbox(f"Primary Key", key=f"pk_{i}")
                    
                    if col_name:
                        columns.append({
                            "name": col_name,
                            "type": col_type,
                            "primary_key": is_primary
                        })
                
                if st.button("Generate CREATE TABLE SQL"):
                    if table_name and columns:
                        column_defs = []
                        for col in columns:
                            col_def = f"{col['name']} {col['type']}"
                            if col['primary_key']:
                                col_def += " PRIMARY KEY"
                            column_defs.append(col_def)
                        
                        # create_sql = f"CREATE TABLE {table_name} (\n    {',\n    '.join(column_defs)}\n);"
                        newline = ",\n    "
                        column_def_str = newline.join(column_defs)
                        create_sql = f"CREATE TABLE {table_name} (\n    {column_def_str}\n);"

                        st.code(create_sql, language="sql")
                        
                        if st.button("Execute CREATE TABLE"):
                            success, result = st.session_state.db_manager.execute_query(create_sql)
                            if success:
                                st.success(result)
                            else:
                                st.error(f"Error: {result}")
            
            elif operation == "Insert Data":
                st.subheader("Insert New Data")
                
                # Get available tables
                table_info = st.session_state.db_manager.get_table_info()
                if table_info:
                    table_names = [t['table'] for t in table_info]
                    selected_table = st.selectbox("Select Table:", table_names)
                    
                    # Get columns for selected table
                    selected_table_info = next((t for t in table_info if t['table'] == selected_table), None)
                    if selected_table_info:
                        st.write(f"Columns in {selected_table}:")
                        
                        values = {}
                        for col_name, col_type in selected_table_info['columns']:
                            if 'INT' in col_type.upper():
                                values[col_name] = st.number_input(f"{col_name} ({col_type}):", key=f"insert_{col_name}")
                            elif 'FLOAT' in col_type.upper() or 'DECIMAL' in col_type.upper():
                                values[col_name] = st.number_input(f"{col_name} ({col_type}):", 
                                                                 value=0.0, format="%.2f", key=f"insert_{col_name}")
                            elif 'DATE' in col_type.upper():
                                values[col_name] = st.date_input(f"{col_name} ({col_type}):", key=f"insert_{col_name}")
                            else:
                                values[col_name] = st.text_input(f"{col_name} ({col_type}):", key=f"insert_{col_name}")
                        
                        if st.button("Generate INSERT SQL"):
                            columns = list(values.keys())
                            formatted_values = []
                            for col in columns:
                                val = values[col]
                                if isinstance(val, str):
                                    formatted_values.append(f"'{val}'")
                                else:
                                    formatted_values.append(str(val))
                            
                            insert_sql = f"INSERT INTO {selected_table} ({', '.join(columns)}) VALUES ({', '.join(formatted_values)});"
                            st.code(insert_sql, language="sql")
                            
                            if st.button("Execute INSERT"):
                                success, result = st.session_state.db_manager.execute_query(insert_sql)
                                if success:
                                    st.success(result)
                                else:
                                    st.error(f"Error: {result}")
            
            elif operation == "Update Data":
                st.subheader("Update Existing Data")
                
                table_info = st.session_state.db_manager.get_table_info()
                if table_info:
                    table_names = [t['table'] for t in table_info]
                    selected_table = st.selectbox("Select Table:", table_names, key="update_table")
                    
                    selected_table_info = next((t for t in table_info if t['table'] == selected_table), None)
                    if selected_table_info:
                        st.write("Set new values:")
                        set_clause = st.text_area("SET clause (e.g., name='John', age=30):", 
                                                 placeholder="column1='value1', column2=value2")
                        
                        where_clause = st.text_input("WHERE clause (e.g., id=1):", 
                                                   placeholder="id=1 (REQUIRED for safety)")
                        
                        if st.button("Generate UPDATE SQL"):
                            if set_clause and where_clause:
                                update_sql = f"UPDATE {selected_table} SET {set_clause} WHERE {where_clause};"
                                st.code(update_sql, language="sql")
                                
                                if st.button("Execute UPDATE"):
                                    success, result = st.session_state.db_manager.execute_query(update_sql)
                                    if success:
                                        st.success(result)
                                    else:
                                        st.error(f"Error: {result}")
                            else:
                                st.warning("Both SET and WHERE clauses are required!")
            
            elif operation == "Delete Data":
                st.subheader("Delete Data")
                st.warning("⚠️ Be careful with DELETE operations!")
                
                table_info = st.session_state.db_manager.get_table_info()
                if table_info:
                    table_names = [t['table'] for t in table_info]
                    selected_table = st.selectbox("Select Table:", table_names, key="delete_table")
                    
                    where_clause = st.text_input("WHERE clause (REQUIRED):", 
                                               placeholder="id=1 (specify which records to delete)")
                    
                    if st.button("Generate DELETE SQL"):
                        if where_clause:
                            delete_sql = f"DELETE FROM {selected_table} WHERE {where_clause};"
                            st.code(delete_sql, language="sql")
                            
                            st.warning(f"This will delete records from {selected_table} where {where_clause}")
                            
                            if st.button("⚠️ Execute DELETE (IRREVERSIBLE)"):
                                success, result = st.session_state.db_manager.execute_query(delete_sql)
                                if success:
                                    st.success(result)
                                else:
                                    st.error(f"Error: {result}")
                        else:
                            st.error("WHERE clause is required for DELETE operations!")
            
            elif operation == "Drop Table":
                st.subheader("Drop Table")
                st.error("⚠️ DANGER: This will permanently delete the entire table and all its data!")
                
                table_info = st.session_state.db_manager.get_table_info()
                if table_info:
                    table_names = [t['table'] for t in table_info]
                    selected_table = st.selectbox("Select Table to Drop:", table_names, key="drop_table")
                    
                    confirmation = st.text_input("Type 'DELETE FOREVER' to confirm:", key="drop_confirm")
                    
                    if confirmation == "DELETE FOREVER":
                        drop_sql = f"DROP TABLE {selected_table};"
                        st.code(drop_sql, language="sql")
                        
                        if st.button("🗑️ DROP TABLE (PERMANENT)"):
                            success, result = st.session_state.db_manager.execute_query(drop_sql)
                            if success:
                                st.success(result)
                                st.rerun()  # Refresh to update table list
                            else:
                                st.error(f"Error: {result}")
                    else:
                        st.info("Type 'DELETE FOREVER' in the confirmation box to enable the drop button.")
                
                else:
                    st.info("No tables available to drop.")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to AI Data Analyst!</h3>
        <p>This application allows you to:</p>
        <ul>
            <li>🤖 Ask questions in natural language and get SQL queries</li>
            <li>🗄️ Connect to multiple database types (SQLite, MySQL, PostgreSQL)</li>
            <li>📊 Explore database schemas and data</li>
            <li>📈 Generate automatic visualizations</li>
            <li>⚡ Get instant insights from your data</li>
        </ul>
        <p><strong>Getting Started:</strong></p>
        <ol>
            <li>Enter your Groq API key in the sidebar</li>
            <li>Choose your database type and connect</li>
            <li>Start asking questions about your data!</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                <h4>🤖 AI-Powered</h4>
                <p>Uses Llama 3 via Groq for fast and accurate SQL generation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                <h4>🗄️ Multi-Database</h4>
                <p>Supports SQLite, MySQL, and PostgreSQL databases</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                <h4>📊 Auto-Visualization</h4>
                <p>Automatically creates charts and graphs from your data</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
