import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

# --- Конфиг ---
app = FastAPI()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lifetrack.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- База данных ---
class Habit(Base):
    __tablename__ = "habits"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    color = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class HabitLog(Base):
    __tablename__ = "habit_logs"
    id = Column(Integer, primary_key=True)
    habit_id = Column(Integer)
    date = Column(DateTime)
    completed = Column(Boolean)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    amount = Column(Float)
    category = Column(String)
    type = Column(String)  # income / expense
    date = Column(DateTime, default=datetime.utcnow)
    note = Column(String, nullable=True)

class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    author = Column(String)
    total_pages = Column(Integer)
    current_page = Column(Integer, default=0)
    status = Column(String)  # reading, completed, want_to_read

class DiaryEntry(Base):
    __tablename__ = "diary"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    mood = Column(String)  # great, good, neutral, sad, bad
    tags = Column(String)  # JSON array
    date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Pydantic схемы ---
class HabitCreate(BaseModel):
    name: str
    color: str = "#3B82F6"

class HabitLogCreate(BaseModel):
    habit_id: int
    date: str
    completed: bool

class TransactionCreate(BaseModel):
    amount: float
    category: str
    type: str
    note: Optional[str] = None

class BookCreate(BaseModel):
    title: str
    author: str
    total_pages: int

class BookProgress(BaseModel):
    current_page: int

class DiaryCreate(BaseModel):
    content: str
    mood: str
    tags: List[str]

class ChatRequest(BaseModel):
    message: str

# --- ИИ помощник ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "YOUR_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def get_user_context():
    db = SessionLocal()
    habits = db.query(Habit).all()
    habit_logs = db.query(HabitLog).filter(HabitLog.date >= datetime.utcnow() - timedelta(days=7)).all()
    transactions = db.query(Transaction).filter(Transaction.date >= datetime.utcnow() - timedelta(days=30)).all()
    books = db.query(Book).all()
    diary = db.query(DiaryEntry).order_by(DiaryEntry.date.desc()).limit(5).all()
    db.close()
    
    context = f"""
    Habits: {[h.name for h in habits]}
    Last 7 days completions: {len([l for l in habit_logs if l.completed])} checkins
    Finances: last 30 days - {sum(t.amount for t in transactions if t.type=='expense')} expenses, {sum(t.amount for t in transactions if t.type=='income')} income
    Books: {len([b for b in books if b.status=='reading'])} reading, {len([b for b in books if b.status=='completed'])} completed
    Latest diary moods: {[e.mood for e in diary]}
    """
    return context

@app.post("/api/chat")
async def chat(request: ChatRequest):
    context = get_user_context()
    prompt = f"You are LifeTrack AI assistant. User data: {context}\nUser: {request.message}\nAI:"
    response = model.generate_content(prompt)
    return {"response": response.text}

# --- API Эндпоинты ---
@app.get("/api/habits")
def get_habits():
    db = SessionLocal()
    habits = db.query(Habit).all()
    db.close()
    return habits

@app.post("/api/habits")
def create_habit(habit: HabitCreate):
    db = SessionLocal()
    db_habit = Habit(name=habit.name, color=habit.color)
    db.add(db_habit)
    db.commit()
    db.refresh(db_habit)
    db.close()
    return db_habit

@app.post("/api/habit-logs")
def log_habit(log: HabitLogCreate):
    db = SessionLocal()
    date_obj = datetime.fromisoformat(log.date)
    existing = db.query(HabitLog).filter(HabitLog.habit_id == log.habit_id, HabitLog.date == date_obj).first()
    if existing:
        existing.completed = log.completed
    else:
        db.add(HabitLog(habit_id=log.habit_id, date=date_obj, completed=log.completed))
    db.commit()
    db.close()
    return {"success": True}

@app.get("/api/habit-streak/{habit_id}")
def get_streak(habit_id: int):
    db = SessionLocal()
    logs = db.query(HabitLog).filter(HabitLog.habit_id == habit_id).order_by(HabitLog.date.desc()).all()
    streak = 0
    today = datetime.utcnow().date()
    for log in logs:
        if log.date.date() == today - timedelta(days=streak) and log.completed:
            streak += 1
        else:
            break
    db.close()
    return {"streak": streak}

@app.get("/api/finances")
def get_finances():
    db = SessionLocal()
    transactions = db.query(Transaction).all()
    db.close()
    return transactions

@app.post("/api/finances")
def add_transaction(trans: TransactionCreate):
    db = SessionLocal()
    db_trans = Transaction(**trans.dict())
    db.add(db_trans)
    db.commit()
    db.refresh(db_trans)
    db.close()
    return db_trans

@app.get("/api/books")
def get_books():
    db = SessionLocal()
    books = db.query(Book).all()
    db.close()
    return books

@app.post("/api/books")
def add_book(book: BookCreate):
    db = SessionLocal()
    db_book = Book(**book.dict(), status="reading")
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    db.close()
    return db_book

@app.put("/api/books/{book_id}/progress")
def update_progress(book_id: int, progress: BookProgress):
    db = SessionLocal()
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(404, "Book not found")
    book.current_page = progress.current_page
    if book.current_page >= book.total_pages:
        book.status = "completed"
    db.commit()
    db.close()
    return {"success": True}

@app.get("/api/diary")
def get_diary():
    db = SessionLocal()
    entries = db.query(DiaryEntry).order_by(DiaryEntry.date.desc()).all()
    db.close()
    return entries

@app.post("/api/diary")
def add_diary(entry: DiaryCreate):
    db = SessionLocal()
    db_entry = DiaryEntry(content=entry.content, mood=entry.mood, tags=json.dumps(entry.tags))
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    db.close()
    return db_entry

# --- Фронтенд (всё в одном HTML) ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LifeTrack — Твой трекер жизни с ИИ</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .habit-card { transition: all 0.2s; }
        .habit-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body class="bg-gray-50">
    <div class="max-w-7xl mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-gray-800 mb-2">🧬 LifeTrack</h1>
        <p class="text-gray-600 mb-8">Привычки · Финансы · Книги · Дневник + ИИ помощник</p>
        
        <!-- Tabs -->
        <div class="flex gap-2 border-b mb-6">
            <button onclick="showTab('habits')" class="tab-btn px-4 py-2 font-semibold text-blue-600 border-b-2 border-blue-600">📊 Привычки</button>
            <button onclick="showTab('finances')" class="tab-btn px-4 py-2 font-semibold text-gray-600">💰 Финансы</button>
            <button onclick="showTab('books')" class="tab-btn px-4 py-2 font-semibold text-gray-600">📚 Книги</button>
            <button onclick="showTab('diary')" class="tab-btn px-4 py-2 font-semibold text-gray-600">📔 Дневник</button>
            <button onclick="showTab('chat')" class="tab-btn px-4 py-2 font-semibold text-gray-600">🤖 ИИ Чат</button>
        </div>
        
        <!-- Habits Tab -->
        <div id="habits" class="tab-content">
            <div class="mb-4 flex gap-2">
                <input type="text" id="habitName" placeholder="Новая привычка" class="border rounded px-3 py-2 flex-1">
                <button onclick="addHabit()" class="bg-blue-500 text-white px-4 py-2 rounded">➕ Добавить</button>
            </div>
            <div id="habitsList" class="grid md:grid-cols-2 gap-4"></div>
        </div>
        
        <!-- Finances Tab -->
        <div id="finances" class="tab-content hidden">
            <div class="grid md:grid-cols-2 gap-6">
                <div>
                    <h3 class="font-bold mb-2">➕ Добавить транзакцию</h3>
                    <input type="number" id="amount" placeholder="Сумма" class="border rounded p-2 w-full mb-2">
                    <input type="text" id="category" placeholder="Категория" class="border rounded p-2 w-full mb-2">
                    <select id="type" class="border rounded p-2 w-full mb-2">
                        <option value="expense">Расход</option>
                        <option value="income">Доход</option>
                    </select>
                    <button onclick="addTransaction()" class="bg-green-500 text-white px-4 py-2 rounded">Добавить</button>
                </div>
                <div>
                    <canvas id="financeChart" width="400" height="300"></canvas>
                </div>
            </div>
            <div id="transactionsList" class="mt-4"></div>
        </div>
        
        <!-- Books Tab -->
        <div id="books" class="tab-content hidden">
            <div class="mb-4 grid md:grid-cols-4 gap-2">
                <input type="text" id="bookTitle" placeholder="Название" class="border rounded p-2">
                <input type="text" id="bookAuthor" placeholder="Автор" class="border rounded p-2">
                <input type="number" id="bookPages" placeholder="Страниц" class="border rounded p-2">
                <button onclick="addBook()" class="bg-purple-500 text-white px-4 py-2 rounded">➕ Добавить</button>
            </div>
            <div id="booksList" class="grid md:grid-cols-3 gap-4"></div>
        </div>
        
        <!-- Diary Tab -->
        <div id="diary" class="tab-content hidden">
            <div class="bg-white rounded-lg shadow p-4 mb-4">
                <textarea id="diaryContent" rows="3" placeholder="Что произошло сегодня?" class="border rounded p-2 w-full"></textarea>
                <div class="flex gap-2 mt-2">
                    <select id="diaryMood" class="border rounded p-2">
                        <option value="great">😍 Отлично</option>
                        <option value="good">😊 Хорошо</option>
                        <option value="neutral">😐 Нормально</option>
                        <option value="sad">😔 Грустно</option>
                        <option value="bad">😫 Плохо</option>
                    </select>
                    <input type="text" id="diaryTags" placeholder="Теги через запятую (работа, спорт...)" class="border rounded p-2 flex-1">
                    <button onclick="addDiaryEntry()" class="bg-indigo-500 text-white px-4 py-2 rounded">Сохранить</button>
                </div>
            </div>
            <div id="diaryEntries"></div>
        </div>
        
        <!-- AI Chat Tab -->
        <div id="chat" class="tab-content hidden">
            <div class="bg-white rounded-lg shadow h-96 overflow-y-auto p-4 mb-4" id="chatMessages">
                <div class="text-gray-500">👋 Привет! Я знаю твои привычки, финансы, книги и дневник. Спроси меня что-нибудь!</div>
            </div>
            <div class="flex gap-2">
                <input type="text" id="chatInput" placeholder="Спроси у ИИ..." class="border rounded p-2 flex-1">
                <button onclick="sendChat()" class="bg-blue-500 text-white px-6 py-2 rounded">Отправить</button>
            </div>
        </div>
    </div>
    
    <script>
        let financeChart = null;
        
        async function apiCall(url, method='GET', data=null) {
            const options = { method, headers: {'Content-Type': 'application/json'} };
            if (data) options.body = JSON.stringify(data);
            const res = await fetch(url, options);
            return res.json();
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
            document.getElementById(tab).classList.remove('hidden');
            if (tab === 'finances') loadFinances();
            if (tab === 'habits') loadHabits();
            if (tab === 'books') loadBooks();
            if (tab === 'diary') loadDiary();
        }
        
        // Habits
        async function loadHabits() {
            const habits = await apiCall('/api/habits');
            const container = document.getElementById('habitsList');
            container.innerHTML = '';
            for (const habit of habits) {
                const streak = await apiCall(`/api/habit-streak/${habit.id}`);
                const today = new Date().toISOString().split('T')[0];
                const log = await apiCall(`/api/habit-logs?habit_id=${habit.id}&date=${today}`).catch(() => null);
                const completed = log?.completed || false;
                container.innerHTML += `
                    <div class="habit-card bg-white rounded-lg shadow p-4 border-l-8" style="border-left-color: ${habit.color}">
                        <div class="flex justify-between items-center">
                            <div>
                                <h3 class="font-bold text-lg">${habit.name}</h3>
                                <p class="text-sm text-gray-500">🔥 Серия: ${streak.streak} дней</p>
                            </div>
                            <button onclick="toggleHabit(${habit.id}, ${!completed})" class="px-4 py-2 rounded ${completed ? 'bg-green-500' : 'bg-gray-300'} text-white">
                                ${completed ? '✅ Выполнено' : '⭕ Отметить'}
                            </button>
                        </div>
                    </div>
                `;
            }
        }
        
        async function addHabit() {
            const name = document.getElementById('habitName').value;
            if (!name) return;
            await apiCall('/api/habits', 'POST', { name });
            loadHabits();
            document.getElementById('habitName').value = '';
        }
        
        async function toggleHabit(habitId, completed) {
            await apiCall('/api/habit-logs', 'POST', { habit_id: habitId, date: new Date().toISOString().split('T')[0], completed });
            loadHabits();
        }
        
        // Finances
        async function loadFinances() {
            const transactions = await apiCall('/api/finances');
            const expenses = transactions.filter(t => t.type === 'expense').reduce((sum, t) => sum + t.amount, 0);
            const income = transactions.filter(t => t.type === 'income').reduce((sum, t) => sum + t.amount, 0);
            if (financeChart) financeChart.destroy();
            const ctx = document.getElementById('financeChart').getContext('2d');
            financeChart = new Chart(ctx, {
                type: 'doughnut',
                data: { labels: ['Расходы', 'Доходы'], datasets: [{ data: [expenses, income], backgroundColor: ['#EF4444', '#10B981'] }] }
            });
            const list = document.getElementById('transactionsList');
            list.innerHTML = '<h3 class="font-bold mb-2">📜 История</h3>' + transactions.map(t => `<div class="flex justify-between border-b py-2"><span>${t.category}</span><span class="${t.type === 'expense' ? 'text-red-600' : 'text-green-600'}">${t.type === 'expense' ? '-' : '+'}${t.amount} ₽</span><span class="text-sm text-gray-500">${new Date(t.date).toLocaleDateString()}</span></div>`).join('');
        }
        
        async function addTransaction() {
            const amount = parseFloat(document.getElementById('amount').value);
            const category = document.getElementById('category').value;
            const type = document.getElementById('type').value;
            if (!amount || !category) return;
            await apiCall('/api/finances', 'POST', { amount, category, type });
            loadFinances();
            document.getElementById('amount').value = '';
            document.getElementById('category').value = '';
        }
        
        // Books
        async function loadBooks() {
            const books = await apiCall('/api/books');
            const container = document.getElementById('booksList');
            container.innerHTML = books.map(book => `
                <div class="bg-white rounded-lg shadow p-4">
                    <h3 class="font-bold">${book.title}</h3>
                    <p class="text-sm text-gray-600">${book.author}</p>
                    <div class="mt-2">
                        <div class="bg-gray-200 rounded-full h-2">
                            <div class="bg-purple-600 rounded-full h-2" style="width: ${(book.current_page/book.total_pages)*100}%"></div>
                        </div>
                        <p class="text-xs mt-1">${book.current_page} / ${book.total_pages} стр.</p>
                        <input type="range" min="0" max="${book.total_pages}" value="${book.current_page}" onchange="updateProgress(${book.id}, this.value)" class="w-full mt-2">
                    </div>
                </div>
            `).join('');
        }
        
        async function addBook() {
            const title = document.getElementById('bookTitle').value;
            const author = document.getElementById('bookAuthor').value;
            const total_pages = parseInt(document.getElementById('bookPages').value);
            if (!title || !author || !total_pages) return;
            await apiCall('/api/books', 'POST', { title, author, total_pages });
            loadBooks();
        }
        
        async function updateProgress(bookId, page) {
            await apiCall(`/api/books/${bookId}/progress`, 'PUT', { current_page: parseInt(page) });
            loadBooks();
        }
        
        // Diary
        async function loadDiary() {
            const entries = await apiCall('/api/diary');
            const container = document.getElementById('diaryEntries');
            container.innerHTML = entries.map(entry => `
                <div class="bg-white rounded-lg shadow p-4 mb-3">
                    <div class="flex justify-between items-start">
                        <span class="text-2xl">${entry.mood === 'great' ? '😍' : entry.mood === 'good' ? '😊' : entry.mood === 'neutral' ? '😐' : entry.mood === 'sad' ? '😔' : '😫'}</span>
                        <span class="text-sm text-gray-500">${new Date(entry.date).toLocaleDateString()}</span>
                    </div>
                    <p class="mt-2">${entry.content}</p>
                    <div class="mt-2 flex gap-2">
                        ${JSON.parse(entry.tags || '[]').map(tag => `<span class="bg-gray-200 px-2 py-1 rounded text-xs">#${tag}</span>`).join('')}
                    </div>
                </div>
            `).join('');
        }
        
        async function addDiaryEntry() {
            const content = document.getElementById('diaryContent').value;
            const mood = document.getElementById('diaryMood').value;
            const tags = document.getElementById('diaryTags').value.split(',').map(t => t.trim()).filter(t => t);
            if (!content) return;
            await apiCall('/api/diary', 'POST', { content, mood, tags });
            loadDiary();
            document.getElementById('diaryContent').value = '';
            document.getElementById('diaryTags').value = '';
        }
        
        // AI Chat
        async function sendChat() {
            const input = document.getElementById('chatInput');
            const message = input.value;
            if (!message) return;
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.innerHTML += `<div class="text-right mb-2"><span class="bg-blue-100 inline-block p-2 rounded-lg">${message}</span></div>`;
            input.value = '';
            const response = await apiCall('/api/chat', 'POST', { message });
            messagesDiv.innerHTML += `<div class="text-left mb-2"><span class="bg-gray-100 inline-block p-2 rounded-lg">🤖 ${response.response}</span></div>`;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        loadHabits();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)