# 🤖 AI Data Analyst

Transform natural language questions into SQL queries and get instant insights from your databases with the power of AI!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![AI Powered](https://img.shields.io/badge/AI-Groq%20LLaMA%203-purple)

## 🌟 Features

- **🧠 Natural Language to SQL**: Convert plain English questions into SQL queries using Groq's LLaMA 3 AI model
- **🗄️ Multi-Database Support**: Connect to SQLite, MySQL, and PostgreSQL databases
- **📊 Automatic Visualizations**: Generate charts and graphs automatically from query results
- **💾 Full CRUD Operations**: Create, Read, Update, and Delete data with an intuitive interface
- **🔍 Database Explorer**: Browse schemas, tables, and columns easily
- **📈 Data Analysis**: Built-in visualization tools with Plotly
- **⚡ Real-time Query Execution**: Execute queries and see results instantly
- **📥 Export Capabilities**: Download query results as CSV files

## 🚀 Demo

### Natural Language Query Example
Ask: "Show me the top 5 employees by salary"

The AI automatically generates:
```sql
SELECT name, salary, department 
FROM employees 
ORDER BY salary DESC 
LIMIT 5
```

### Supported Query Types
- Data retrieval: "What are the total sales by region?"
- Aggregations: "Show me average salary by department"
- Joins: "List all employees with their sales totals"
- Data modifications: "Update John's salary to 80000"
- Schema operations: "Create a new table for products"

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (get one from [Groq Console](https://console.groq.com))
- Database server (for MySQL/PostgreSQL)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-data-analyst.git
cd ai-data-analyst
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
pandas>=2.0.0
sqlite3
mysql-connector-python>=8.0.0
psycopg2-binary>=2.9.0
pymongo>=4.0.0
sqlalchemy>=2.0.0
pymysql>=1.0.0
plotly>=5.0.0
langchain>=0.1.0
langchain-groq>=0.0.1
langchain-community>=0.0.1
```

## 🏃‍♂️ Quick Start

1. **Run the application**
```bash
streamlit run app.py
```

2. **Open your browser**
Navigate to `http://localhost:8501`

3. **Configure your setup**
   - Enter your Groq API key in the sidebar
   - Choose your database type
   - Connect to your database or create a sample one

4. **Start querying!**
   - Use natural language to ask questions about your data
   - Execute SQL queries directly
   - Visualize results instantly

## 🗄️ Database Configuration

### SQLite
- Create a new sample database with demo data
- Upload an existing `.db`, `.sqlite`, or `.sqlite3` file

### MySQL
```python
Host: localhost
Port: 3306
Username: your_username
Password: your_password
Database: your_database_name
```

### PostgreSQL
```python
Host: localhost
Port: 5432
Username: your_username
Password: your_password
Database: your_database_name
```

## 💡 Usage Examples

### 1. Natural Language Queries
- "Show me all employees earning more than $70,000"
- "What's the average sale amount by region?"
- "List the top 10 products by revenue"
- "How many employees work in each department?"

### 2. Direct SQL Queries
```sql
SELECT e.name, e.department, SUM(s.amount) as total_sales
FROM employees e
JOIN sales s ON e.id = s.employee_id
GROUP BY e.id, e.name, e.department
ORDER BY total_sales DESC;
```

### 3. CRUD Operations
- **Create**: Design and create new tables with custom schemas
- **Insert**: Add new records with form-based input
- **Update**: Modify existing data with safety checks
- **Delete**: Remove records with confirmation

## 🎨 Features in Detail

### AI-Powered SQL Generation
- Uses Groq's LLaMA 3 70B model for accurate SQL generation
- Context-aware query generation based on your database schema
- Handles complex queries including JOINs, aggregations, and subqueries

### Visual Database Explorer
- Browse all tables and their structures
- View column names and data types
- Preview sample data from each table

### Automatic Visualizations
- Bar charts for categorical comparisons
- Line charts for trends over time
- Scatter plots for correlations
- Pie charts for distributions

### Safety Features
- Required WHERE clauses for UPDATE/DELETE operations
- Confirmation prompts for destructive operations
- Transaction support for data integrity

## 🔧 Configuration Options

### Environment Variables (Optional)
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
DEFAULT_DB_TYPE=sqlite
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

## 🐛 Troubleshooting

### Common Issues

1. **MySQL Connection Error**
   - Ensure MySQL server is running
   - Check credentials and database exists
   - Verify port 3306 is not blocked

2. **Groq API Key Invalid**
   - Get a valid API key from [Groq Console](https://console.groq.com)
   - Check for trailing spaces in the API key

3. **Module Not Found**
   - Run `pip install -r requirements.txt`
   - Ensure virtual environment is activated

### Debug Mode
Add to your code for detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com) for providing the LLaMA 3 API
- [Streamlit](https://streamlit.io) for the amazing web framework
- [LangChain](https://langchain.com) for LLM integration tools
- [Plotly](https://plotly.com) for interactive visualizations

## 📞 Support

- Create an issue for bug reports or feature requests
- Star ⭐ the repository if you find it helpful
- Follow for updates and new features

## 🚀 Future Enhancements

- [ ] Support for more database types (Oracle, MongoDB)
- [ ] Advanced data profiling and statistics
- [ ] Export to multiple formats (Excel, JSON, Parquet)
- [ ] Scheduled query execution
- [ ] Query history and saved queries
- [ ] Collaborative features
- [ ] API endpoint for programmatic access

---

Made with ❤️ by [Your Name]

**Note**: This is an educational project demonstrating AI-powered data analysis. Always review generated SQL queries before execution in production environments.
