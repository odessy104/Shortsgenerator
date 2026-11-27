import os
import glob
import random
import asyncio
import re
import time
import json
import requests
import warnings
import pickle
import difflib
import wave
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import edge_tts
import whisper

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

from moviepy import (
    VideoFileClip, 
    AudioFileClip, 
    ImageClip, 
    CompositeVideoClip, 
    CompositeAudioClip, 
    concatenate_videoclips
)

# Google / YouTube API Importe
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# ⚙️ KONFIGURATION & UMGEBUNGSERKENNUNG
# ==========================================
AUTO_UPLOAD = False  # True = Automatischer YouTube-Upload | False = Nur lokales Rendern
TARGET_FRAMES = 900  # Exakt 30 Sekunden (30 FPS)
FPS = 30
TARGET_DURATION = TARGET_FRAMES / FPS

# Erkennen, ob das Skript in GitHub Actions (Online) oder lokal läuft
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

# Dynamische Font-Konfiguration je nach Umgebung
if IS_GITHUB_ACTIONS:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(FONT_PATH):
        FONT_PATH = "DejaVuSans-Bold"
    SUBTITLE_FONT_SIZE = 65
    print("☁️ Umgebung: GitHub Actions (Online) – Verwende Cloud-Schriftgröße 65 & Wikimedia.")
else:
    FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists(FONT_PATH):
        FONT_PATH = "arial.ttf"
    SUBTITLE_FONT_SIZE = 48
    print("💻 Umgebung: Lokaler Laptop – Verwende lokale Schriftgröße 48 & DDGS.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

FALLBACK_EMOJI_MAP = {
    "WELTALL": "🌌", "RAUM": "🚀", "ZEIT": "⏳", "GEFAHR": "⚠️", 
    "UNIVERSUM": "✨", "EXPLOSION": "💥", "LICHT": "💡", "ERDE": "🌍", 
    "SONNE": "☀️", "GRAVITATION": "🌀", "ENDE": "🏁", "GEHEIMNIS": "🔍", 
    "WISSENSCHAFT": "🔬", "GEHIRN": "🧠", "MASSE": "🪐", "MENSCH": "👤",
    "LOCH": "🕳️", "LÖCHER": "🕳️", "SCHWARZES": "🕳️", "KERN": "🔥", "SCHWERELOSIGKEIT": "🛸",
    "TIEFSEE": "🌊", "OZEAN": "🌊", "SCHLAF": "😴", "STURM": "⚡", "ERINNERUNG": "🧠",
    "MANUSKRIPT": "📜", "BUCH": "📖", "SCHRIFT": "✍️"
}

def setup_directories():
    if os.path.exists("result.json"):
        try:
            os.remove("result.json")
            print("🗑️ Alte result.json automatisch zurückgesetzt.")
        except Exception as e:
            print(f"⚠️ Konnte result.json nicht löschen: {e}")

    os.makedirs("input_clips", exist_ok=True)
    os.makedirs("background_music", exist_ok=True)
    os.makedirs("sfx", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    ensure_builtin_cinematic_audio()

def update_pipeline_status(progress_pct, current_step_text):
    """Schreibt den aktuellen Fortschritt in eine status.json für die Web-App."""
    status_data = {
        "running": True,
        "progress": progress_pct,
        "message": current_step_text,
        "timestamp": time.time()
    }
    try:
        with open("status.json", "w", encoding="utf-8") as f:
            json.dump(status_data, f, ensure_ascii=False)
    except Exception:
        pass

def ensure_builtin_cinematic_audio():
    sample_rate = 44100
    ambient_path = "background_music/builtin_cinematic_ambient.wav"
    if not os.path.exists(ambient_path):
        duration = 32.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        pad = (
            0.35 * np.sin(2 * np.pi * 73.42 * t) +
            0.25 * np.sin(2 * np.pi * 110.00 * t) +
            0.18 * np.sin(2 * np.pi * 174.61 * t + 0.3 * np.sin(2 * np.pi * 0.12 * t)) +
            0.12 * np.sin(2 * np.pi * 220.00 * t + 0.2 * np.sin(2 * np.pi * 0.08 * t)) +
            0.08 * np.sin(2 * np.pi * 329.63 * t)
        )
        lfo = 0.78 + 0.22 * np.sin(2 * np.pi * 0.15 * t)
        pad = pad * lfo
        fade_samples = int(sample_rate * 1.5)
        pad[:fade_samples] *= np.linspace(0, 1, fade_samples)
        pad[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        pad = pad / (np.max(np.abs(pad)) + 1e-6) * 0.85
        audio_int16 = np.int16(pad * 32767)
        stereo = np.column_stack((audio_int16, audio_int16))
        with wave.open(ambient_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(stereo.tobytes())

    whoosh_path = "sfx/builtin_soft_whoosh.wav"
    if not os.path.exists(whoosh_path):
        w_dur = 0.65
        t_w = np.linspace(0, w_dur, int(sample_rate * w_dur), endpoint=False)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1, len(t_w))
        kernel = np.ones(45) / 45.0
        soft_noise = np.convolve(noise, kernel, mode="same")
        soft_noise = np.convolve(soft_noise, kernel, mode="same")
        freq_sweep = np.linspace(95, 42, len(t_w))
        sub_body = 0.45 * np.sin(2 * np.pi * np.cumsum(freq_sweep) / sample_rate)
        envelope = np.sin(np.pi * (t_w / w_dur)) ** 2.2
        whoosh = (soft_noise * 2.8 + sub_body) * envelope
        whoosh = whoosh / (np.max(np.abs(whoosh)) + 1e-6) * 0.85
        w_int16 = np.int16(whoosh * 32767)
        w_stereo = np.column_stack((w_int16, w_int16))
        with wave.open(whoosh_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(w_stereo.tobytes())

def extract_json_from_response(content):
    try:
        return json.loads(content)
    except Exception:
        pass

    cleaned = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = cleaned[start_idx:end_idx+1]
            return json.loads(potential_json)
    except Exception:
        pass

    return None

def proofread_german_grammar(parsed_json, headers):
    try:
        parts = parsed_json.get("parts", [])
        if not parts:
            return parsed_json
        review_payload = {
            "viral_title": parsed_json.get("viral_title", ""),
            "texts": [p.get("text", "") for p in parts]
        }
        proof_system = (
            "Du bist ein kompromissloser deutscher Chef-Lektor für High-End-Dokumentationen (Duden-Niveau). "
            "Prüfe die Sätze streng auf Grammatik, Eleganz und natürlichen Redefluss. "
            "VERBANNE unnatürliche Wortkonstruktionen wie 'erreicht einen Fakt' oder 'verlangsamt sich das Wissen'. Nutze echte, flüssige Doku-Formulierungen. "
            "BEWAHRE alle XML-Tags wie <emphasis level=\"strong\">WORT</emphasis>, Emojis und die exakte Satzanzahl bei! "
            "Antworte AUSSCHLIESSLICH als reines JSON-Objekt."
        )
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": proof_system},
                {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        res = requests.post(GROQ_URL, json=payload, headers=headers, timeout=20)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            fixed = extract_json_from_response(content)
            if fixed:
                fixed_texts = fixed.get("texts", [])
                if len(fixed_texts) == len(parts):
                    for idx, new_t in enumerate(fixed_texts):
                        if new_t and isinstance(new_t, str):
                            parsed_json["parts"][idx]["text"] = new_t
                if fixed.get("viral_title"):
                    parsed_json["viral_title"] = fixed["viral_title"]
                print("🔍 Cheflektorat: Sätze perfektioniert & natürlich geformt!")
    except Exception as e:
        print(f"⚠️ Lektorat-Hinweis: {e}")
    return parsed_json

def intelligent_web_fallback_script(topic):
    print(f"🌐 Suche echte Web-Fakten für das Fallback-Skript zu '{topic}'...")
    snippets = []
    if DDGS:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{topic} fakten wissenschaft hintergrund", max_results=5))
                for r in results:
                    body = r.get("body", "")
                    if body:
                        snippets.append(body)
        except Exception:
            pass

    fact_sentences = []
    if snippets:
        combined_text = " ".join(snippets)
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        for s in sentences:
            clean_s = s.strip()
            if len(clean_s) > 35 and len(clean_s) < 100 and not any(w in clean_s.lower() for w in ["cookie", "datenschutz", "werbung"]):
                fact_sentences.append(clean_s)

    while len(fact_sentences) < 5:
        fact_sentences.append(f"Wissenschaftler erforschen dieses komplexe Phänomen rund um {topic} seit vielen Jahren intensiv 🔬.")

    emojis = ["🧐", "⚡", "🔬", "🌍", "💡", "🚀"]
    parts = []
    
    parts.append({
        "text": f'Kaum jemand ahnt, welches faszinierende Geheimnis hinter <emphasis level="strong">GEHEIMNIS</emphasis> steckt 🕵️‍♂️.',
        "search_keyword": f"{topic} macro scientific landscape photography"
    })
    
    for i in range(4):
        sent = fact_sentences[i] if i < len(fact_sentences) else f"Das erstaunliche Phänomen von {topic} stellt Forscher weltweit vor ungelöste Fragen ⚡."
        if len(sent) > 85:
            sent = sent[:82] + "..."
        parts.append({
            "text": f"{sent} {emojis[i % len(emojis)]}",
            "search_keyword": f"{topic} scientific macro nature background"
        })
        
    parts.append({
        "text": f'Bis heute bleibt die gesamte Wissenschaft über das große Rätsel von <emphasis level="strong">RÄTSEL</emphasis> sprachlos 🌍.',
        "search_keyword": f"{topic} cinematic nature wide shot"
    })

    return {
        "mood": "epic",
        "viral_title": f"{topic} – Die ungelöste Akte #shorts",
        "parts": parts
    }

def generate_full_script_with_cloud_ai(topic_or_text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    print(f"⚡ Groq ({GROQ_MODEL}) kreiert das Skript und die Fakten zum Thema: '{topic_or_text}'...")
    
    system_prompt = (
        "Du bist ein preisgekrönter Drehbuchautor für hochwertige wissenschaftliche Dokumentationen.\n"
        "Schreibe ein fesselndes, präzises Skript auf Deutsch, das gesprochen exakt 26 bis 28 Sekunden lang ist (insgesamt ca. 55 bis 65 Wörter – halte es bewusst etwas kompakter, damit es nicht zu gehetzt wirkt!).\n"
        "STIL: Verwende absolut natürliches, elegantes Deutsch im Arte/Netflix-Dokumentationsstil. Vermeide hölzerne KI-Formulierungen wie 'erreicht einen Fakt' völlig.\n"
        "WICHTIG für 'viral_title': Erstelle einen extrem klickstarken, seriösen Doku-Titel mit #shorts am Ende. Keine billigen Clickbait-Floskeln.\n"
        "WICHTIG für Emphasis-Tags: Setze in <emphasis level=\"strong\">...</emphasis> immer ein echtes, wichtiges Hauptwort ein (z.B. MYSTERIUM oder KERN) und NIEMALS das Wort 'WORT'.\n"
        "WICHTIG für 'search_keyword': Generiere präzise, eindeutige ENGLISCHE Suchbegriffe, die direkt zum genauen Thema passen (z.B. für das Gedächtnis: 'human brain memory neuroscience visualization', für Tiere: 'animal superpower wildlife macro photography'). Schließe Menschen, Sportler, Hochzeiten und Food strikt aus.\n"
        "Antworte AUSSCHLIESSLICH als valides JSON-Objekt im exakten Format unten:\n\n"
        "{\n"
        "  \"mood\": \"epic\",\n"
        "  \"viral_title\": \"Dein starker Doku-Titel #shorts\",\n"
        "  \"parts\": [\n"
        "    {\"text\": \"Hook mit <emphasis level=\\\"strong\\\">GEHEIMNIS</emphasis> (max. 10 Wörter)\", \"search_keyword\": \"english specific scientific query 1\"},\n"
        "    {\"text\": \"Fakt mit <emphasis level=\\\"strong\\\">KERN</emphasis> (max. 10 Wörter)\", \"search_keyword\": \"english specific scientific query 2\"},\n"
        "    {\"text\": \"Fakt mit <emphasis level=\\\"strong\\\">PHÄNOMEN</emphasis> (max. 10 Wörter)\", \"search_keyword\": \"english specific scientific query 3\"},\n"
        "    {\"text\": \"Fakt mit <emphasis level=\\\"strong\\\">FORSCHUNG</emphasis> (max. 10 Wörter)\", \"search_keyword\": \"english specific scientific query 4\"},\n"
        "    {\"text\": \"Fakt mit <emphasis level=\\\"strong\\\">ENERGIE</emphasis> (max. 10 Wörter)\", \"search_keyword\": \"english specific scientific query 5\"},\n"
        "    {\"text\": \"Cliffhanger mit <emphasis level=\\\"strong\\\">RÄTSEL</emphasis> (max. 10 Wörter)\", \"search_keyword\": \"english specific scientific query 6\"}\n"
        "  ]\n"
        "}"
    )

    user_prompt = f"Thema: {topic_or_text}. Erstelle das 6-Teile JSON exakt zu diesem spezifischen Thema im Doku-Stil."
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 4096
    }
    
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=25)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_json_from_response(content)
            if parsed and "parts" in parsed and len(parsed["parts"]) >= 4:
                parsed = proofread_german_grammar(parsed, headers)
                print(f"✨ Skript erfolgreich generiert und optimiert!")
                return parsed
    except Exception as e:
        print(f"⚠️ Groq-Exception: {e}")
        
    return intelligent_web_fallback_script(topic_or_text)

def extract_highlights_and_emojis(parts):
    highlight_set = set()
    emoji_map = FALLBACK_EMOJI_MAP.copy()
    for part in parts:
        text = part.get("text", "")
        matches = re.findall(r'<emphasis level="strong">(.*?)</emphasis>', text)
        for m in matches:
            clean_m = re.sub(r'[^A-Za-zÄÖÜäöüß0-9]', '', m).upper()
            if clean_m:
                highlight_set.add(clean_m)
        tokens = text.split()
        for i in range(len(tokens) - 1):
            w = re.sub(r'[^A-Za-zÄÖÜäöüß0-9]', '', tokens[i]).upper()
            next_t = tokens[i+1]
            if w and any(ord(c) > 127 for c in next_t):
                emoji_map[w] = next_t
    return highlight_set, emoji_map

def get_topics_queue():
    themen_env = os.getenv("THEMEN_LISTE", "").strip()
    if themen_env:
        topics = [t.strip() for t in themen_env.split(",") if t.strip()]
        if topics:
            return topics

    script_path = "skript.txt"
    if not os.path.exists(script_path):
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("Warum Uhren im Weltall langsamer oder schneller ticken\n")
    with open(script_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines if lines else ["Warum Uhren im Weltall langsamer oder schneller ticken"]

def is_url_blocked_watermark_or_thumbnail(url, title=""):
    u = (url + " " + title).lower()
    blocked_sources = [
        "alamy", "shutterstock", "istockphoto", "gettyimages", "dreamstime",
        "123rf", "depositphotos", "vecteezy", "stock.adobe", "bigstockphoto",
        "plus.unsplash", "ytimg.com", "youtube.com", "youtu.be", "tiktok.com",
        "pinterest.", "fbsbx.com", "fbsbx", "slideplayer", "slideshare",
        "cosplay", "deviantart", "artstation", "pinterest", "instagram", "imgur", "anime",
        "netflix", "amazon", "book", "cover", "poster", "banner", "logo", "shop",
        "food", "recipe", "dish", "cooking", "kitchen", "restaurant", "eat", "menu",
        "capcom", "boardgame", "miniature", "ad", "promo",
        "wedding", "bride", "groom", "ceremony", "marriage", "people", "portrait", "face", "crowd",
        "soccer", "football", "player", "sport", "match", "stadium", "athlete", "team",
        "league", "goal", "ball", "championship", "tournament", "jersey", "coach", "celebrity", "actor"
    ]
    return any(b in u for b in blocked_sources)

def is_image_clean_and_cinematic(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if w < 800 or h < 600:
            return False
        aspect = w / float(h)
        if aspect > 2.4 or aspect < 0.4:
            return False
        pixels = np.array(img, dtype=np.float32)
        if (np.mean(pixels[0:25, :, :]) > 225 or np.mean(pixels[-25:, :, :]) > 225 or 
            np.mean(pixels[:, 0:25, :]) > 225 or np.mean(pixels[:, -25:, :]) > 225):
            return False
        if np.std(pixels) < 22.0 or np.mean(pixels) < 15.0 or np.mean(pixels) > 210.0:
            return False
        return True
    except Exception:
        return False

def search_ddgs_image_pool(queries_to_try, used_urls):
    if not DDGS:
        return None
    for q in queries_to_try:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(q, max_results=20))
                for item in results:
                    img_url = item.get("image", "")
                    source_url = item.get("url", "")
                    title = item.get("title", "").lower()
                    if not img_url or not isinstance(img_url, str) or not img_url.startswith("http"):
                        continue
                    if img_url in used_urls:
                        continue
                    if is_url_blocked_watermark_or_thumbnail(img_url, title + " " + source_url):
                        continue
                    try:
                        resp = requests.get(img_url, timeout=4, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                        img_bytes = resp.content
                        if len(img_bytes) > 60000 and is_image_clean_and_cinematic(img_bytes):
                            test_img = Image.open(BytesIO(img_bytes))
                            test_img.verify()
                            used_urls.add(img_url)
                            print(f"   📸 Perfektes DDGS-Bild gefunden: '{q}'")
                            return img_bytes
                    except Exception:
                        continue
        except Exception:
            time.sleep(0.3)
            continue
    return None

def search_wikimedia_image_pool(query, main_topic, used_urls):
    search_url = "[https://commons.wikimedia.org/w/api.php](https://commons.wikimedia.org/w/api.php)"
    clean_q = re.sub(r'\b(macro|photography|wide shot|angle|realistic|cinematic|background|high resolution|scientific|visualization)\b', '', query, flags=re.IGNORECASE).strip()
    words = clean_q.split()
    short_q = " ".join(words[:3]) if len(words) > 3 else clean_q
    
    search_queries = [short_q, f"{main_topic} {words[0] if words else ''}".strip(), main_topic]
    
    headers = {
        "User-Agent": "ScienceShortsGenerator/1.0 ([https://github.com](https://github.com); contact@scienceshorts.local) Python-Requests"
    }
    
    for sq in search_queries:
        if not sq:
            continue
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {sq}",
            "gsrnamespace": "6",
            "gsrlimit": "12",
            "prop": "imageinfo",
            "iiprop": "url|size",
            "iiurlwidth": "1280",
            "format": "json"
        }
        try:
            time.sleep(0.6)
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200 or not response.text.strip():
                continue
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page_info in pages.items():
                title = page_info.get("title", "")
                image_info = page_info.get("imageinfo", [])
                if image_info:
                    info = image_info[0]
                    img_url = info.get("thumburl") or info.get("url")
                    if not img_url or img_url in used_urls:
                        continue
                    if is_url_blocked_watermark_or_thumbnail(img_url, title):
                        continue
                    try:
                        time.sleep(0.3)
                        resp = requests.get(img_url, timeout=8, headers=headers)
                        if resp.status_code != 200:
                            continue
                        img_bytes = resp.content
                        if len(img_bytes) > 40000 and is_image_clean_and_cinematic(img_bytes):
                            test_img = Image.open(BytesIO(img_bytes))
                            test_img.verify()
                            used_urls.add(img_url)
                            print(f"   📸 Perfektes Wikimedia-Bild gefunden: '{sq}'")
                            return img_bytes
                    except Exception:
                        continue
        except Exception as e:
            print(f"   ⚠️ Wikimedia-Hinweis: {e}")
            time.sleep(1.0)
    return None

def download_contextual_media(parts, main_topic):
    for f_path in glob.glob("input_clips/*"):
        filename = os.path.basename(f_path)
        if filename.startswith("cloud_media_") or "_perfect_fit" in f_path or "_sub_temp" in f_path or "_outro_temp" in f_path:
            try:
                os.remove(f_path)
            except:
                pass

    downloaded_files = []
    used_urls = set()
    strict_negatives = "-soccer -football -player -sport -match -stadium -athlete -team -wedding -people -portrait -food -recipe"

    for idx, part in enumerate(parts):
        ai_query = part.get('search_keyword', main_topic)
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', ai_query).strip()
        if not clean_q:
            clean_q = main_topic
        
        saved_path = None
        img_bytes = None
        
        if IS_GITHUB_ACTIONS:
            print(f"☁️ [GitHub] Bildsuche {idx+1}/6 über Wikimedia für Thema '{main_topic}' / Keyword: '{clean_q}'...")
            img_bytes = search_wikimedia_image_pool(clean_q, main_topic, used_urls)
        else:
            print(f"💻 [Lokal] Bildsuche {idx+1}/6 über DDGS für: '{clean_q}'...")
            queries = [
                f"{clean_q} scientific macro photography wide shot realistic {strict_negatives}",
                f"{main_topic} documentary scientific background high resolution {strict_negatives}"
            ]
            img_bytes = search_ddgs_image_pool(queries, used_urls)
            
        if not img_bytes and IS_GITHUB_ACTIONS and DDGS:
            print("   🔄 Wikimedia lieferte kein Trefferbild, versuche DDGS...")
            queries = [
                f"{clean_q} scientific macro photography {strict_negatives}",
                f"{main_topic} nature background {strict_negatives}"
            ]
            img_bytes = search_ddgs_image_pool(queries, used_urls)
            
        if not img_bytes:
            print("   ⚠️ Offline-Fallback: Generiere Cinematic-Hintergrund...")
            img_blank = Image.new("RGB", (1080, 1920), (15, 18, 28))
            draw = ImageDraw.Draw(img_blank)
            draw.ellipse([-200, -200, 1280, 800], fill=(30, 45, 75))
            buf = BytesIO()
            img_blank.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

        if img_bytes:
            saved_path = f"input_clips/cloud_media_{idx+1}_{random.randint(1000,9999)}.jpg"
            with open(saved_path, "wb") as f:
                f.write(img_bytes)
                
        if saved_path and os.path.exists(saved_path):
            downloaded_files.append(saved_path)
            
    return downloaded_files

def create_perfect_background_clip(image_path, target_duration=3.0):
    pil_img = Image.open(image_path).convert('RGB')
    target_w, target_h = 1080, 1920
    img_w, img_h = pil_img.size

    bg_scale = max(target_w / img_w, target_h / img_h)
    bg_w = int(img_w * bg_scale)
    bg_h = int(img_h * bg_scale)
    bg_resized = pil_img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
    bx1 = (bg_w - target_w) // 2
    by1 = (bg_h - target_h) // 2
    bg_cropped = bg_resized.crop((bx1, by1, bx1 + target_w, by1 + target_h))
    
    enhancer = ImageEnhance.Brightness(bg_cropped)
    bg_dark = enhancer.enhance(0.35)

    fg_scale = min(target_w / img_w, target_h / img_h) * 0.90
    fg_w = int(img_w * fg_scale)
    fg_h = int(img_h * fg_scale)
    fg_resized = pil_img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (10, 12, 18))
    fx = (target_w - fg_w) // 2
    fy = (target_h - fg_h) // 2
    
    canvas.paste(bg_dark, (0, 0))
    canvas.paste(fg_resized, (fx, fy))

    temp_path = image_path + f"_perfect_fit_{random.randint(100,999)}.jpg"
    canvas.save(temp_path, "JPEG", quality=95)
    return ImageClip(temp_path).with_duration(target_duration).with_position('center')

def make_perfect_timeline(media_clips, total_duration):
    clips_sequence = []
    if media_clips:
        exact_seg_dur = total_duration / len(media_clips)
        for clip in media_clips:
            dur = clip.duration
            if dur < exact_seg_dur:
                loops = int(exact_seg_dur // dur) + 1
                clip_looped = concatenate_videoclips([clip] * loops).subclipped(0, exact_seg_dur)
            else:
                clip_looped = clip.subclipped(0, exact_seg_dur)
            
            sub = clip_looped.resized(lambda t: 1.0 + 0.03 * (t / max(0.1, clip_looped.duration)))
            sub = sub.with_position('center')
            clips_sequence.append(sub)
    if not clips_sequence:
        return None
    final_timeline = concatenate_videoclips(clips_sequence, method="compose")
    return final_timeline.subclipped(0, min(total_duration, final_timeline.duration)).with_position('center')

def create_pil_text_image(text_block, font_path, emoji_map, dynamic_highlights, font_size=48, canvas_width=1000):
    words = text_block.split()
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    word_gap = 14
    
    lines = []
    current_line_words = []
    current_line_width = 0
    max_line_width = 900
    
    for word in words:
        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]
        if current_line_width + w + (len(current_line_words) * word_gap) > max_line_width and current_line_words:
            lines.append(current_line_words)
            current_line_words = [word]
            current_line_width = w
        else:
            current_line_words.append(word)
            current_line_width += w
            
    if current_line_words:
        lines.append(current_line_words)
        
    line_height = font_size + 12
    img_h = len(lines) * line_height + 30
    img = Image.new("RGBA", (canvas_width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    stroke_width = 4
    y = 10
    for line_words in lines:
        line_w = 0
        line_word_data = []
        for word in line_words:
            clean_w = re.sub(r'[^A-Za-zÄÖÜäöüß0-9]', '', word).upper()
            is_highlight = clean_w in dynamic_highlights
            bbox = draw.textbbox((0, 0), word, font=font)
            w = bbox[2] - bbox[0]
            line_word_data.append({"word": word, "width": w, "highlight": is_highlight})
            line_w += w
        line_w += (len(line_words) - 1) * word_gap
        
        current_x = (canvas_width - line_w) // 2
        for item in line_word_data:
            w_text = item["word"]
            w_width = item["width"]
            is_hl = item["highlight"]
            
            for adj_x in range(-stroke_width, stroke_width + 1):
                for adj_y in range(-stroke_width, stroke_width + 1):
                    if adj_x**2 + adj_y**2 <= stroke_width**2:
                        draw.text((current_x + adj_x, y + adj_y), w_text, font=font, fill="black")
            
            text_color = (255, 225, 0) if is_hl else (255, 255, 255)
            draw.text((current_x, y), w_text, font=font, fill=text_color)
            current_x += w_width + word_gap
            
        y += line_height
        
    temp_path = f"input_clips/_sub_temp_{random.randint(10000, 99999)}.png"
    img.save(temp_path)
    return temp_path

def align_whisper_words_with_script(whisper_words, true_script_text):
    true_tokens = [t for t in true_script_text.split() if re.search(r'[A-Za-zÄÖÜäöüß0-9]', t)]
    if not whisper_words or not true_tokens:
        return whisper_words
    w_norm = [re.sub(r'[^a-zäöüß0-9]', '', w["word"].lower()) for w in whisper_words]
    t_norm = [re.sub(r'[^a-zäöüß0-9]', '', t.lower()) for t in true_tokens]
    matcher = difflib.SequenceMatcher(None, w_norm, t_norm)
    corrected = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for wi, ti in zip(range(i1, i2), range(j1, j2)):
                item = whisper_words[wi].copy()
                item["word"] = true_tokens[ti]
                corrected.append(item)
        elif tag == 'replace':
            span_w = whisper_words[i1:i2]
            span_t = true_tokens[j1:j2]
            if span_w and span_t:
                start_t = span_w[0]["start"]
                end_t = span_w[-1]["end"]
                step = (end_t - start_t) / max(1, len(span_t))
                for k, tok in enumerate(span_t):
                    corrected.append({
                        "word": tok,
                        "start": start_t + k * step,
                        "end": start_t + (k + 1) * step
                    })
        elif tag == 'delete':
            corrected.extend(whisper_words[i1:i2])
        elif tag == 'insert':
            ref_t = whisper_words[i1]["start"] if i1 < len(whisper_words) else (whisper_words[-1]["end"] if whisper_words else 0.0)
            for k, tok in enumerate(true_tokens[j1:j2]):
                corrected.append({
                    "word": tok,
                    "start": ref_t,
                    "end": ref_t + 0.25
                })
    return corrected

def generate_synchronized_subtitles(audio_path, video_size, emoji_map, dynamic_highlights, spoken_script_hint="", max_duration=TARGET_DURATION):
    print("📝 Synchronisiere Untertitel...")
    model = whisper.load_model("small")
    try:
        result = model.transcribe(audio_path, language="de", word_timestamps=True, condition_on_previous_text=False, fp16=False)
    except Exception:
        result = model.transcribe(audio_path, language="de", word_timestamps=False, condition_on_previous_text=False, fp16=False)
    raw_whisper_words = []
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        if words:
            for w_info in words:
                raw_w = w_info.get("word", "").strip()
                if raw_w:
                    raw_whisper_words.append({
                        "word": raw_w,
                        "start": w_info["start"],
                        "end": w_info["end"]
                    })
    aligned_words = align_whisper_words_with_script(raw_whisper_words, spoken_script_hint) if raw_whisper_words else []
    chunks = []
    if aligned_words:
        current_chunk_words = []
        chunk_start = 0.0
        chunk_end = 0.0
        for w_info in aligned_words:
            raw_w = w_info["word"]
            if not current_chunk_words:
                chunk_start = w_info["start"]
            current_chunk_words.append(raw_w)
            chunk_end = w_info["end"]
            is_ordinal_date = bool(re.match(r'^\d+\.$', raw_w))
            has_sentence_end = any(p in raw_w for p in [".", "!", "?", ","]) and not is_ordinal_date
            if (len(current_chunk_words) >= 4 and not is_ordinal_date) or has_sentence_end:
                chunks.append({
                    "text": " ".join(current_chunk_words).upper(),
                    "start": chunk_start,
                    "end": chunk_end
                })
                current_chunk_words = []
        if current_chunk_words:
            chunks.append({
                "text": " ".join(current_chunk_words).upper(),
                "start": chunk_start,
                "end": chunk_end
            })
    else:
        for segment in result.get("segments", []):
            seg_text = segment.get("text", "").strip()
            if seg_text:
                chunks.append({
                    "text": seg_text.upper(),
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 1)
                })
    if chunks and chunks[0]["start"] > 1.5:
        chunks[0]["start"] = 0.0
    subtitle_clips = []
    w, h = video_size

    for i in range(len(chunks)):
        chunk = chunks[i]
        start = chunk["start"]
        if start >= max_duration:
            break
        if i < len(chunks) - 1:
            next_start = chunks[i + 1]["start"]
            end = min(max(chunk["end"], next_start), max_duration)
            if end <= start:
                end = min(start + 0.35, max_duration)
        else:
            end = min(max(start + 0.5, chunk["end"]), max_duration)
        duration = max(0.3, end - start)
        
        img_path = create_pil_text_image(chunk["text"], FONT_PATH, emoji_map, dynamic_highlights, font_size=SUBTITLE_FONT_SIZE, canvas_width=1000)
        txt_clip = (ImageClip(img_path)
                    .with_duration(duration)
                    .with_start(start)
                    .with_position(('center', h * 0.74)))
        subtitle_clips.append(txt_clip)
    print(f"✅ {len(subtitle_clips)} Untertitel synchronisiert!")
    return subtitle_clips

def remove_green_screen(get_frame, t):
    frame = get_frame(t)
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    mask = (g > 80) & (g > r + 30) & (g > b + 30)
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    return np.dstack((frame, alpha))

def get_outro_clip(video_size, duration=3.5, start_time=0.0):
    w, h = video_size
    outro_path = "input_clips/outro.mp4"
    if os.path.exists(outro_path):
        try:
            clip = VideoFileClip(outro_path)
            clip = clip.transform(remove_green_screen, keep_duration=True)
            clip = clip.resized(width=int(w * 0.85))
            return clip.subclipped(0, min(duration, clip.duration)).with_start(start_time).with_position(('center', h * 0.40))
        except Exception:
            pass

    try:
        font = ImageFont.truetype(FONT_PATH, 48)
        small_font = ImageFont.truetype(FONT_PATH, 34)
    except:
        font = ImageFont.load_default()
        small_font = font

    img_w, img_h = 880, 240
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, img_w, img_h], radius=28, fill=(15, 18, 26, 235), outline=(255, 225, 0, 220), width=4)
    draw.text((65, 42), "👍 LIKE & ABONNIEREN", font=font, fill=(255, 225, 0, 255))
    draw.text((65, 132), "Für mehr tägliche Doku-Fakten!", font=small_font, fill=(255, 255, 255, 255))
    temp_path = f"input_clips/_outro_temp_{random.randint(1000, 9999)}.png"
    img.save(temp_path)
    return ImageClip(temp_path).with_duration(duration).with_start(start_time).with_position(('center', h * 0.42))

def upload_short_to_youtube(video_path, title, description="#shorts #science #funfacts"):
    SCOPES = ["[https://www.googleapis.com/auth/youtube.upload](https://www.googleapis.com/auth/youtube.upload)"]
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("client_secret.json"):
                return
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    try:
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": ["shorts", "science", "funfacts", "wissenschaft"],
                "categoryId": "27"
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False
            }
        }
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = request.next_chunk()
        print(f"✅ Hochgeladen! Video-ID: {response.get('id')}")
    except Exception as e:
        print(f"⚠️ YouTube-Upload Fehler: {e}")

def clean_text_for_smooth_speech(raw_text):
    text_clean = re.sub(r'<emphasis[^>]*>(.*?)</emphasis>', r'\1', raw_text)
    text_clean = re.sub(r'<[^>]+>', ' ', text_clean)
    text_clean = re.sub(r'\bWORT\b', '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r'Teil\s*\d+\s*[:,-]?', '', text_clean, flags=re.IGNORECASE)
    
    text_clean = re.sub(r'\[[^\]]*\]', ' ', text_clean)
    text_clean = re.sub(r'\b(SFX|ACTION)_[A-Z_]+\b', ' ', text_clean)
    text_clean = ''.join(c for c in text_clean if c.isalnum() or c.isspace() or c in ".,!?;:-–„\"''()")
    text_clean = text_clean.replace("_", " ")
    text_clean = re.sub(r'\.{2,}', '.', text_clean)
    text_clean = text_clean.replace(";", ",").replace(" – ", ", ").replace(" - ", ", ")
    text_clean = re.sub(r'\s+([.,!?:])', r'\1', text_clean)
    text_clean = re.sub(r'([.,!?:])\s*([.,!?:])+', r'\1', text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    return text_clean

def trim_long_silences(audio_path, max_pause_sec=0.14):
    try:
        clip = AudioFileClip(audio_path)
        fps = clip.fps
        samples = clip.to_soundarray(fps=fps)
        clip.close()
        if len(samples) == 0:
            return
        mono = np.max(np.abs(samples), axis=1) if samples.ndim > 1 else np.abs(samples)
        threshold = 0.015
        min_silent_samples = int(fps * max_pause_sec)
        keep_silent_samples = int(fps * 0.09)
        is_silent = mono < threshold
        result_chunks = []
        i = 0
        n = len(samples)
        while i < n:
            if is_silent[i]:
                j = i
                while j < n and is_silent[j]:
                    j += 1
                silent_len = j - i
                if i == 0:
                    pass
                elif silent_len > min_silent_samples:
                    result_chunks.append(samples[i:i + keep_silent_samples])
                else:
                    result_chunks.append(samples[i:j])
                i = j
            else:
                j = i
                while j < n and not is_silent[j]:
                    j += 1
                result_chunks.append(samples[i:j])
                i = j
        if result_chunks:
            trimmed = np.concatenate(result_chunks, axis=0)
            trimmed_int16 = np.int16(np.clip(trimmed, -1.0, 1.0) * 32767)
            temp_wav = audio_path + ".trimmed.wav"
            channels = trimmed_int16.shape[1] if trimmed_int16.ndim > 1 else 1
            with wave.open(temp_wav, "w") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(fps)
                wf.writeframes(trimmed_int16.tobytes())
            new_clip = AudioFileClip(temp_wav)
            new_clip.write_audiofile(audio_path, logger=None)
            new_clip.close()
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    except Exception:
        pass

async def generate_audio_async(full_text, output_file="voiceover.mp3"):
    print("🎙️ Generiere Sprachausgabe...")
    voice = "de-DE-KillianNeural"
    text_clean = clean_text_for_smooth_speech(full_text)
    if not text_clean:
        text_clean = "Kein Text verfügbar."
    
    current_rate = -2
    communicate = edge_tts.Communicate(text_clean, voice, rate=f"{current_rate:+d}%")
    await communicate.save(output_file)
    trim_long_silences(output_file, max_pause_sec=0.14)
    
    temp_clip = AudioFileClip(output_file)
    actual_dur = temp_clip.duration
    temp_clip.close()
    
    min_target_dur = TARGET_DURATION - 3.5
    max_target_dur = TARGET_DURATION - 2.0
    
    for attempt in range(2):
        if min_target_dur <= actual_dur <= max_target_dur:
            break
        ideal_dur = TARGET_DURATION - 2.5
        ratio = actual_dur / ideal_dur
        current_multiplier = 1.0 + (current_rate / 100.0)
        new_multiplier = current_multiplier * ratio
        new_rate = int(round((new_multiplier - 1.0) * 100.0))
        new_rate = max(-10, min(4, new_rate))
        if new_rate == current_rate:
            break
        current_rate = new_rate
        rate_str = f"{current_rate:+d}%"
        communicate = edge_tts.Communicate(text_clean, voice, rate=rate_str)
        await communicate.save(output_file)
        trim_long_silences(output_file, max_pause_sec=0.14)
        check_clip = AudioFileClip(output_file)
        actual_dur = check_clip.duration
        check_clip.close()
        
    print(f"✅ Sprachausgabe beendet bei {actual_dur:.2f}s!")
    return text_clean

def render_short(script_data, topic, spoken_hint="", index_num=1):
    viral_title = script_data.get("viral_title", f"{topic} #shorts")
    parts = script_data.get("parts", [])
    dynamic_highlights, dynamic_emoji_map = extract_highlights_and_emojis(parts)
    
    downloaded_media_paths = download_contextual_media(parts, topic)
    
    voice_audio = AudioFileClip("voiceover.mp3")
    if voice_audio.duration > TARGET_DURATION:
        voice_audio = voice_audio.subclipped(0, TARGET_DURATION)
    final_duration = TARGET_DURATION
    
    music_paths = glob.glob("background_music/*.mp3") + glob.glob("background_music/*.wav")
    custom_music = [m for m in music_paths if "builtin_cinematic_ambient" not in m]
    chosen_music = random.choice(custom_music) if custom_music else (music_paths[0] if music_paths else None)
    bg_audio = None
    if chosen_music and os.path.exists(chosen_music):
        bg_audio = AudioFileClip(chosen_music)
        if bg_audio.duration < final_duration:
            loops = int(final_duration // bg_audio.duration) + 1
            bg_audio = concatenate_videoclips([bg_audio] * loops).audio
        bg_audio = bg_audio.subclipped(0, final_duration).with_volume_scaled(0.11)
        
    sfx_paths = glob.glob("sfx/*.mp3") + glob.glob("sfx/*.wav")
    custom_sfx = [s for s in sfx_paths if "builtin_soft_whoosh" not in s]
    chosen_whoosh = random.choice(custom_sfx) if custom_sfx else (sfx_paths[0] if sfx_paths else None)
    sfx_clips = []
    outro_duration = 3.5
    outro_start_time = max(0, final_duration - outro_duration)
    if chosen_whoosh and os.path.exists(chosen_whoosh):
        w_intro = AudioFileClip(chosen_whoosh).with_start(0.0).with_volume_scaled(0.18)
        w_outro = AudioFileClip(chosen_whoosh).with_start(max(0.0, outro_start_time - 0.15)).with_volume_scaled(0.16)
        sfx_clips.extend([w_intro, w_outro])
        
    subtitles = generate_synchronized_subtitles("voiceover.mp3", (1080, 1920), dynamic_emoji_map, dynamic_highlights, spoken_script_hint=spoken_hint, max_duration=final_duration)
    
    audio_mix_list = [voice_audio]
    if bg_audio:
        audio_mix_list.append(bg_audio)
    audio_mix_list.extend(sfx_clips)
    final_audio = CompositeAudioClip(audio_mix_list).with_duration(final_duration)
    
    media_clip_objects = []
    for mp in downloaded_media_paths:
        try:
            img_clip = create_perfect_background_clip(mp, target_duration=3.0)
            media_clip_objects.append(img_clip)
        except Exception:
            pass
            
    if not media_clip_objects:
        return
        
    background_video = make_perfect_timeline(media_clip_objects, final_duration)
    background_video = background_video.with_audio(final_audio)
    
    outro_clip = get_outro_clip((1080, 1920), duration=outro_duration, start_time=outro_start_time)
    final_video = CompositeVideoClip([background_video] + subtitles + [outro_clip], size=(1080, 1920)).with_duration(final_duration)
    
    output_path = f"output/fertiges_short_{index_num}.mp4"
    print(f"🎬 Rendere Short #{index_num} mit exakt {TARGET_FRAMES} Frames ({final_duration}s)...")
    final_video.write_videofile(
        output_path, 
        fps=FPS, 
        codec="libx264", 
        audio_codec="aac", 
        pixel_format="yuv420p",
        preset="fast",
        bitrate="5M"
    )
    
    for c in [voice_audio, bg_audio] + sfx_clips + [outro_clip] + media_clip_objects:
        if c:
            try:
                c.close()
            except:
                pass
    try:
        background_video.close()
        final_video.close()
    except:
        pass
   
    results_list = []
    if os.path.exists("result.json"):
        try:
            with open("result.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    results_list = data
                elif isinstance(data, dict) and "videos" in data:
                    results_list = data["videos"]
        except:
            pass

    results_list.append({
        "title": viral_title,
        "index": index_num,
        "tag": f"short-{index_num}",
        "status": "fertig"
    })

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump({"videos": results_list}, f, ensure_ascii=False, indent=4)

    print(f"\n🚀 BOOM! Short #{index_num} ist fertig: {output_path}")
    print(f"\n📋 DEIN TITEL:\n{viral_title}\n")
    if AUTO_UPLOAD:
        upload_short_to_youtube(output_path, viral_title)

if __name__ == "__main__":
    setup_directories()
    topics = get_topics_queue()
    total_topics = len(topics)
    
    for idx, topic in enumerate(topics, 1):
        prozent = int((idx / total_topics) * 100)
        status_text = f"Generiere Short {idx} von {total_topics} ({prozent}%): {topic}"
        
        # Status für die Web-App live speichern (auch beim Schließen sichtbar)
        update_pipeline_status(prozent, status_text)
        
        print(f"\n================ 🚀 FORTSCHRITT: {prozent}% (Short {idx} von {total_topics}) ================")
        print(f"📌 Aktuelles Thema: '{topic}'")
        
        script_data = generate_full_script_with_cloud_ai(topic)
        full_text = " ".join([p["text"] for p in script_data["parts"]])
        clean_spoken = asyncio.run(generate_audio_async(full_text))
        render_short(script_data, topic, spoken_hint=clean_spoken, index_num=idx)
        
    # Abschluss-Status in status.json schreiben
    try:
        with open("status.json", "w", encoding="utf-8") as f:
            json.dump({"running": False, "progress": 100, "message": "Alle Shorts erfolgreich generiert!"}, f)
    except:
        pass

    print(f"\n🎉 Alle {total_topics} Shorts erfolgreich generiert! (100%)")
