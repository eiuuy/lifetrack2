import os
import json
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

# --- Конфиг ---
app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lifetrack.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- Модели БД ---
class Habit(Base):
    __tablename__ = "habits"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    color = Column(String, default="#3B82F6")
    created_at = Column(DateTime, default=datetime.utcnow)

class HabitLog(Base):
    __tablename__ = "habit_logs"
    id = Column(Integer, primary_key=True)
    habit_id = Column(Integer)
    date = Column(DateTime)
    completed = Column(Boolean, default=False)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    amount = Column(Float)
    category = Column(String)
    type = Column(String)
    date = Column(DateTime, default=datetime.utcnow)
    note = Column(String, nullable=True)

class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    author = Column(String)
    total_pages = Column(Integer)
    current_page = Column(Integer, default=0)
    status = Column(String, default="reading")

class DiaryEntry(Base):
    __tablename__ = "diary"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    mood = Column(String)
    tags = Column(String, default="[]")
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
    tags: List[str] = []

class ChatRequest(BaseModel):
    message: str

# --- ИИ помощник ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    model = None

def get_user_context():
    db = SessionLocal()
    habits = db.query(Habit).all()
    transactions = db.query(Transaction).filter(Transaction.date >= datetime.utcnow() - timedelta(days=30)).all()
    books = db.query(Book).all()
    diary = db.query(DiaryEntry).order_by(DiaryEntry.date.desc()).limit(5).all()
    db.close()
    
    return f"""
    Habits: {[h.name for h in habits]}
    Finances (last 30d): Expenses={sum(t.amount for t in transactions if t.type=='expense')}, Income={sum(t.amount for t in transactions if t.type=='income')}
    Books: Reading={len([b for b in books if b.status=='reading'])}, Completed={len([b for b in books if b.status=='completed'])}
    Recent moods: {[e.mood for e in diary]}
    """

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not model:
        return {"response": "⚠️ Gemini API ключ не настроен. Добавь переменную GEMINI_API_KEY в Render."}
    
    context = get_user_context()
    prompt = f"You are LifeTrack AI. User data: {context}\nUser: {request.message}\nAI:"
    try:
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Ошибка ИИ: {str(e)}"}

# --- API эндпоинты ---
@app.get("/api/habits")
def get_habits():
    db = SessionLocal()
    habits = db.query(Habit).all()
    db.close()
    return [{"id": h.id, "name": h.name, "color": h.color} for h in habits]

@app.post("/api/habits")
def create_habit(habit: HabitCreate):
    db = SessionLocal()
    db_habit = Habit(name=habit.name, color=habit.color)
    db.add(db_habit)
    db.commit()
    db.refresh(db_habit)
    db.close()
    return {"id": db_habit.id, "name": db_habit.name, "color": db_habit.color}

@app.post("/api/habit-logs")
def log_habit(log: HabitLogCreate):
    db = SessionLocal()
    date_obj = datetime.fromisoformat(log.date)
    existing = db.query(HabitLog).filter(
        HabitLog.habit_id == log.habit_id, 
        HabitLog.date == date_obj
    ).first()
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

@app.get("/api/habit-logs")
def get_habit_logs(habit_id: int, date: str):
    db = SessionLocal()
    date_obj = datetime.fromisoformat(date)
    log = db.query(HabitLog).filter(HabitLog.habit_id == habit_id, HabitLog.date == date_obj).first()
    db.close()
    return {"completed": log.completed if log else False}

@app.get("/api/finances")
def get_finances():
    db = SessionLocal()
    transactions = db.query(Transaction).all()
    db.close()
    return [{"id": t.id, "amount": t.amount, "category": t.category, "type": t.type, "date": t.date.isoformat()} for t in transactions]

@app.post("/api/finances")
def add_transaction(trans: TransactionCreate):
    db = SessionLocal()
    db_trans = Transaction(**trans.dict())
    db.add(db_trans)
    db.commit()
    db.refresh(db_trans)
    db.close()
    return {"id": db_trans.id}

@app.get("/api/books")
def get_books():
    db = SessionLocal()
    books = db.query(Book).all()
    db.close()
    return [{"id": b.id, "title": b.title, "author": b.author, "total_pages": b.total_pages, "current_page": b.current_page, "status": b.status} for b in books]

@app.post("/api/books")
def add_book(book: BookCreate):
    db = SessionLocal()
    db_book = Book(**book.dict())
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    db.close()
    return {"id": db_book.id}

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
    return [{"id": e.id, "content": e.content, "mood": e.mood, "tags": json.loads(e.tags), "date": e.date.isoformat()} for e in entries]

@app.post("/api/diary")
def add_diary(entry: DiaryCreate):
    db = SessionLocal()
    db_entry = DiaryEntry(
        content=entry.content, 
        mood=entry.mood, 
        tags=json.dumps(entry.tags)
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    db.close()
    return {"id": db_entry.id}

# --- Фронтенд с экранированными символами ---
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LifeTrack - Track your life with AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        .habit-card { transition: all 0.2s; }
        .habit-card:hover { transform: translateY(-2px); }
        .tab-active { border-bottom-width: 2px; color: #2563eb; border-color: #2563eb; }
    </style>
</head>
<body class="bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen">
    <div class="max-w-6xl mx-auto px-4 py-8">
        <div class="text-center mb-8">
            <h1 class="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">LifeTrack</h1>
            <p class="text-gray-600 mt-2">Habits · Finance · Books · Journal + AI Assistant</p>
        </div>
        
        <div class="flex flex-wrap gap-1 border-b mb-6 bg-white rounded-t-lg px-2">
            <button onclick="showTab('habits')" class="tab-btn px-5 py-3 font-semibold text-gray-600 hover:text-blue-600 transition" data-tab="habits">Habits</button>
            <button onclick="showTab('finances')" class="tab-btn px-5 py-3 font-semibold text-gray-600 hover:text-blue-600 transition" data-tab="finances">Finance</button>
            <button onclick="showTab('books')" class="tab-btn px-5 py-3 font-semibold text-gray-600 hover:text-blue-600 transition" data-tab="books">Books</button>
            <button onclick="showTab('diary')" class="tab-btn px-5 py-3 font-semibold text-gray-600 hover:text-blue-600 transition" data-tab="diary">Journal</button>
            <button onclick="showTab('chat')" class="tab-btn px-5 py-3 font-semibold text-gray-600 hover:text-blue-600 transition" data-tab="chat">AI Chat</button>
        </div>
        
        <div id="habits" class="tab-content">
            <div class="bg-white rounded-xl shadow-md p-5 mb-5">
                <div class="flex gap-3">
                    <input type="text" id="habitName" placeholder="New habit (e.g., Meditation)" class="border rounded-lg px-4 py-2 flex-1 focus:outline-none focus:ring-2 focus:ring-blue-400">
                    <button onclick="addHabit()" class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition">+ Add</button>
                </div>
            </div>
            <div id="habitsList" class="grid md:grid-cols-2 gap-4"></div>
        </div>
        
        <div id="finances" class="tab-content hidden">
            <div class="grid md:grid-cols-2 gap-6">
                <div class="bg-white rounded-xl shadow-md p-5">
                    <h3 class="font-bold text-lg mb-3">+ New Transaction</h3>
                    <input type="number" id="amount" placeholder="Amount" class="border rounded-lg p-2 w-full mb-2">
                    <input type="text" id="category" placeholder="Category (food, transport...)" class="border rounded-lg p-2 w-full mb-2">
                    <select id="type" class="border rounded-lg p-2 w-full mb-3">
                        <option value="expense">Expense</option>
                        <option value="income">Income</option>
                    </select>
                    <button onclick="addTransaction()" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg w-full transition">Add</button>
                </div>
                <div class="bg-white rounded-xl shadow-md p-5">
                    <canvas id="financeChart" height="250"></canvas>
                </div>
            </div>
            <div id="transactionsList" class="bg-white rounded-xl shadow-md p-5 mt-5"></div>
        </div>
        
        <div id="books" class="tab-content hidden">
            <div class="bg-white rounded-xl shadow-md p-5 mb-5">
                <div class="grid md:grid-cols-4 gap-3">
                    <input type="text" id="bookTitle" placeholder="Title" class="border rounded-lg p-2">
                    <input type="text" id="bookAuthor" placeholder="Author" class="border rounded-lg p-2">
                    <input type="number" id="bookPages" placeholder="Pages" class="border rounded-lg p-2">
                    <button onclick="addBook()" class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition">+ Add Book</button>
                </div>
            </div>
            <div id="booksList" class="grid md:grid-cols-3 gap-4"></div>
        </div>
        
        <div id="diary" class="tab-content hidden">
            <div class="bg-white rounded-xl shadow-md p-5 mb-5">
                <textarea id="diaryContent" rows="3" placeholder="What happened today?" class="border rounded-lg p-3 w-full"></textarea>
                <div class="flex gap-3 mt-3">
                    <select id="diaryMood" class="border rounded-lg p-2">
                        <option value="great">Great</option>
                        <option value="good">Good</option>
                        <option value="neutral">Neutral</option>
                        <option value="sad">Sad</option>
                        <option value="bad">Bad</option>
                    </select>
                    <input type="text" id="diaryTags" placeholder="Tags (work, sports...)" class="border rounded-lg p-2 flex-1">
                    <button onclick="addDiaryEntry()" class="bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-2 rounded-lg transition">Save</button>
                </div>
            </div>
            <div id="diaryEntries"></div>
        </div>
        
        <div id="chat" class="tab-content hidden">
            <div class="bg-white rounded-xl shadow-md h-96 overflow-y-auto p-4 mb-4" id="chatMessages">
                <div class="text-gray-500 text-center">Hi! I know your habits, finances, books, and journal. Ask me anything!</div>
            </div>
            <div class="flex gap-3">
                <input type="text" id="chatInput" placeholder="Type your message..." class="border rounded-lg p-3 flex-1 focus:outline-none focus:ring-2 focus:ring-blue-400">
                <button onclick="sendChat()" class="bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-lg transition">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        let financeChart = null;
        
        async function apiCall(url, method='GET', data=null) {
            const options = { method, headers: {'Content-Type': 'application/json'} };
            if (data) options.body = JSON.stringify(data);
            const res = await fetch(url, options);
            if (!res.ok) throw new Error(await res.text());
            return res.json();
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
            document.getElementById(tab).classList.remove('hidden');
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('tab-active', 'text-blue-600', 'border-blue-600');
                btn.classList.add('text-gray-600');
            });
            const activeBtn = document.querySelector(`[data-tab="${tab}"]`) || document.querySelector(`button[onclick="showTab('${tab}')"]`);
            if (activeBtn) {
                activeBtn.classList.add('tab-active', 'text-blue-600', 'border-blue-600');
                activeBtn.classList.remove('text-gray-600');
            }
            if (tab === 'habits') loadHabits();
            if (tab === 'finances') loadFinances();
            if (tab === 'books') loadBooks();
            if (tab === 'diary') loadDiary();
        }
        
        async function loadHabits() {
            const habits = await apiCall('/api/habits');
            const container = document.getElementById('habitsList');
            container.innerHTML = '';
            for (const habit of habits) {
                const streak = await apiCall(`/api/habit-streak/${habit.id}`);
                const today = new Date().toISOString().split('T')[0];
                const log = await apiCall(`/api/habit-logs?habit_id=${habit.id}&date=${today}`);
                container.innerHTML += `
                    <div class="habit-card bg-white rounded-xl shadow-md p-4 border-l-8" style="border-left-color: ${habit.color}">
                        <div class="flex justify-between items-center">
                            <div>
                                <h3 class="font-bold text-lg">${escapeHtml(habit.name)}</h3>
                                <p class="text-sm text-gray-500">Streak: ${streak.streak} days</p>
                            </div>
                            <button onclick="toggleHabit(${habit.id}, ${!log.completed})" class="px-4 py-2 rounded-lg transition ${log.completed ? 'bg-green-500 hover:bg-green-600' : 'bg-gray-400 hover:bg-gray-500'} text-white">
                                ${log.completed ? 'Done' : 'Mark'}
                            </button>
                        </div>
                    </div>
                `;
            }
        }
        
        async function addHabit() {
            const name = document.getElementById('habitName').value.trim();
            if (!name) return alert('Enter habit name');
            await apiCall('/api/habits', 'POST', { name });
            loadHabits();
            document.getElementById('habitName').value = '';
        }
        
        async function toggleHabit(habitId, completed) {
            await apiCall('/api/habit-logs', 'POST', { 
                habit_id: habitId, 
                date: new Date().toISOString().split('T')[0], 
                completed 
            });
            loadHabits();
        }
        
        async function loadFinances() {
            const transactions = await apiCall('/api/finances');
            const expenses = transactions.filter(t => t.type === 'expense').reduce((sum, t) => sum + t.amount, 0);
            const income = transactions.filter(t => t.type === 'income').reduce((sum, t) => sum + t.amount, 0);
            if (financeChart) financeChart.destroy();
            const ctx = document.getElementById('financeChart').getContext('2d');
            financeChart = new Chart(ctx, {
                type: 'doughnut',
                data: { labels: ['Expenses', 'Income'], datasets: [{ data: [expenses, income], backgroundColor: ['#ef4444', '#10b981'] }] },
                options: { responsive: true, maintainAspectRatio: true }
            });
            const list = document.getElementById('transactionsList');
            list.innerHTML = '<h3 class="font-bold mb-3">History</h3>' + 
                (transactions.length === 0 ? '<p class="text-gray-500">No transactions yet</p>' :
                transactions.map(t => `<div class="flex justify-between border-b py-2"><span>${escapeHtml(t.category)}</span><span class="${t.type === 'expense' ? 'text-red-600' : 'text-green-600'} font-semibold">${t.type === 'expense' ? '-' : '+'}${t.amount}</span><span class="text-sm text-gray-500">${new Date(t.date).toLocaleDateString()}</span></div>`).join(''));
        }
        
        async function addTransaction() {
            const amount = parseFloat(document.getElementById('amount').value);
            const category = document.getElementById('category').value.trim();
            const type = document.getElementById('type').value;
            if (!amount || !category) return alert('Fill amount and category');
            await apiCall('/api/finances', 'POST', { amount, category, type });
            loadFinances();
            document.getElementById('amount').value = '';
            document.getElementById('category').value = '';
        }
        
        async function loadBooks() {
            const books = await apiCall('/api/books');
            const container = document.getElementById('booksList');
            if (books.length === 0) {
                container.innerHTML = '<div class="col-span-3 text-center text-gray-500 py-8">Add your first book</div>';
                return;
            }
            container.innerHTML = books.map(book => {
                const percent = (book.current_page / book.total_pages) * 100;
                return `
                    <div class="bg-white rounded-xl shadow-md p-4">
                        <h3 class="font-bold text-lg">${escapeHtml(book.title)}</h3>
                        <p class="text-sm text-gray-600 mb-2">${escapeHtml(book.author)}</p>
                        <div class="mt-3">
                            <div class="bg-gray-200 rounded-full h-2 overflow-hidden">
                                <div class="bg-purple-600 rounded-full h-2 transition-all" style="width: ${percent}%"></div>
                            </div>
                            <div class="flex justify-between text-sm mt-2">
                                <span>${book.current_page} / ${book.total_pages} pages</span>
                                <span>${Math.round(percent)}%</span>
                            </div>
                            <input type="range" min="0" max="${book.total_pages}" value="${book.current_page}" onchange="updateProgress(${book.id}, this.value)" class="w-full mt-3 accent-purple-600">
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        async function addBook() {
            const title = document.getElementById('bookTitle').value.trim();
            const author = document.getElementById('bookAuthor').value.trim();
            const total_pages = parseInt(document.getElementById('bookPages').value);
            if (!title || !author || !total_pages) return alert('Fill all fields');
            await apiCall('/api/books', 'POST', { title, author, total_pages });
            loadBooks();
            document.getElementById('bookTitle').value = '';
            document.getElementById('bookAuthor').value = '';
            document.getElementById('bookPages').value = '';
        }
        
        async function updateProgress(bookId, page) {
            await apiCall(`/api/books/${bookId}/progress`, 'PUT', { current_page: parseInt(page) });
            loadBooks();
        }
        
        async function loadDiary() {
            const entries = await apiCall('/api/diary');
            const container = document.getElementById('diaryEntries');
            if (entries.length === 0) {
                container.innerHTML = '<div class="bg-white rounded-xl shadow-md p-8 text-center text-gray-500">No entries yet. Write something!</div>';
                return;
            }
            container.innerHTML = entries.map(entry => {
                const moodText = { great: 'Great', good: 'Good', neutral: 'Neutral', sad: 'Sad', bad: 'Bad' }[entry.mood] || 'Neutral';
                const tags = Array.isArray(entry.tags) ? entry.tags : [];
                return `
                    <div class="bg-white rounded-xl shadow-md p-4 mb-3">
                        <div class="flex justify-between items-start mb-2">
                            <span class="font-semibold text-gray-700">${moodText}</span>
                            <span class="text-sm text-gray-500">${new Date(entry.date).toLocaleDateString()}</span>
                        </div>
                        <p class="text-gray-800 whitespace-pre-wrap">${escapeHtml(entry.content)}</p>
                        ${tags.length ? `<div class="mt-3 flex flex-wrap gap-2">${tags.map(tag => `<span class="bg-gray-100 px-2 py-1 rounded-lg text-xs text-gray-600">#${escapeHtml(tag)}</span>`).join('')}</div>` : ''}
                    </div>
                `;
            }).join('');
        }
        
        async function addDiaryEntry() {
            const content = document.getElementById('diaryContent').value.trim();
            const mood = document.getElementById('diaryMood').value;
            const tags = document.getElementById('diaryTags').value.split(',').map(t => t.trim()).filter(t => t);
            if (!content) return alert('Write something');
            await apiCall('/api/diary', 'POST', { content, mood, tags });
            loadDiary();
            document.getElementById('diaryContent').value = '';
            document.getElementById('diaryTags').value = '';
        }
        
        async function sendChat() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.innerHTML += '<div class="text-right mb-3"><span class="bg-blue-500 text-white inline-block p-3 rounded-2xl rounded-tr-sm max-w-[80%]">' + escapeHtml(message) + '</span></div>';
            input.value = '';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            try {
                const response = await apiCall('/api/chat', 'POST', { message });
                messagesDiv.innerHTML += '<div class="text-left mb-3"><span class="bg-gray-100 text-gray-800 inline-block p-3 rounded-2xl rounded-tl-sm max-w-[80%]">' + escapeHtml(response.response) + '</span></div>';
            } catch (e) {
                messagesDiv.innerHTML += '<div class="text-left mb-3"><span class="bg-red-100 text-red-600 inline-block p-3 rounded-2xl">Error: ' + escapeHtml(e.message) + '</span></div>';
            }
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/[&<>]/g, function(m) {
                if (m === '&') return '&amp;';
                if (m === '<') return '&lt;';
                if (m === '>') return '&gt;';
                return m;
            });
        }
        
        showTab('habits');
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_TEMPLATE

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
