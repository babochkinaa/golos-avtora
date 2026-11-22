#!/usr/bin/env python3
"""
Корпусный анализ авторского стиля (русский язык).
Часть skill «golos-avtora». Признаки ИИ-текста берутся из slop_check.py (гайд Википедии).

Использование:
    python3 style_analyzer.py <папка_с_текстами> [--out report_dir]

Регистр текста определяется по имени верхней подпапки:
    продажи/ , sales/            -> "продающий"
    устное/ , stories/, transcripts/ -> "устный"
    всё остальное                -> "повседневный"
Поддерживаются: Telegram result.json, .txt, .md, .docx

На выходе: report.md (человекочитаемый отчёт) и profile.json (цифры для skill).
"""
import sys, os, re, json, argparse, statistics as st
from collections import Counter, defaultdict

from razdel import sentenize, tokenize
import pymorphy3

MORPH = pymorphy3.MorphAnalyzer()

# ---------- загрузка ----------

def register_of(path, root):
    rel = os.path.relpath(path, root).lower()
    top = rel.split(os.sep)[0]
    if any(k in top for k in ("прода", "sales", "прогрев", "запуск", "лендинг")):
        return "продающий"
    if any(k in top for k in ("устн", "stories", "сториз", "transcript", "расшифр", "кружк")):
        return "устный"
    return "повседневный"


def tg_text(t):
    if isinstance(t, str):
        return t
    out = []
    for part in t:
        out.append(part if isinstance(part, str) else part.get("text", ""))
    return "".join(out)


NOISE = re.compile(r"Not included, change data exporting settings to download\.?|https?://\S+|t\.me/\S+|@\w+", re.I)


def clean_text(t):
    return re.sub(r"[ \t]{2,}", " ", NOISE.sub("", t)).strip()


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slop_check

def ai_score(t):
    """Индекс нейрослопа (признаки гайда Википедии, см. slop_check.py) на 100 слов."""
    return slop_check.check(t)["score_per_100_words"]


def load_corpus(root):
    docs = []
    has_jsonl = any(f.endswith(".jsonl") for _, _, fs in os.walk(root) for f in fs)
    for dp, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dp, fn)
            reg = register_of(p, root)
            low = fn.lower()
            try:
                if low.endswith(".jsonl"):
                    for line in open(p, encoding="utf-8"):
                        m = json.loads(line)
                        docs.append({"text": m["text"], "register": m.get("register", reg), "src": fn,
                                     "date": m.get("date", ""), "reactions": m.get("reactions"),
                                     "media": m.get("media", [])})
                    continue
                if low.endswith(".json"):
                    data = json.load(open(p, encoding="utf-8"))
                    msgs = data.get("messages", []) if isinstance(data, dict) else []
                    for m in msgs:
                        if m.get("type") != "message":
                            continue
                        txt = tg_text(m.get("text", "")).strip()
                        if len(txt.split()) >= 5:
                            docs.append({"text": txt, "register": reg, "src": fn,
                                         "date": m.get("date", "")})
                elif low.endswith((".txt", ".md")):
                    if has_jsonl and reg != "устный":
                        continue
                    raw = open(p, encoding="utf-8", errors="ignore").read()
                    # несколько постов в одном файле разделяются строкой ---
                    for chunk in re.split(r"\n-{3,}\n", raw):
                        if len(chunk.split()) >= 5:
                            docs.append({"text": chunk.strip(), "register": reg, "src": fn})
                elif low.endswith(".docx"):
                    import docx
                    d = docx.Document(p)
                    txt = "\n".join(par.text for par in d.paragraphs)
                    if len(txt.split()) >= 5:
                        docs.append({"text": txt, "register": reg, "src": fn})
            except Exception as e:
                print(f"[skip] {p}: {e}", file=sys.stderr)
    for d in docs:
        if d["register"] != "устный":
            d["text"] = clean_text(d["text"])
        d["ai_score"] = ai_score(d["text"]) if d["register"] != "устный" else 0
    return [d for d in docs if len(d["text"].split()) >= 5]

# ---------- базовые единицы ----------

WORD_RE = re.compile(r"^[а-яёa-z\-]+$", re.I)
CYR_RE = re.compile(r"^[а-яё\-]+$", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤]")


def words(s):
    return [t.text for t in tokenize(s) if WORD_RE.match(t.text)]


LIST_LINE = re.compile(r"^\s*([✅✔•\-–\*▪️🔹🔸]|\d+[.)]|[\U0001F300-\U0001FAFF\u2600-\u27BF]\uFE0F?\s*\*\*|\*\*[^*]+\*\*\s*:?\s*$)")
SERVICE_LINE = re.compile(r"^\s*(Photo|Video|Sticker|Voice message|\d+×\d+.*KB|\d+×\d+.*MB)\s*$", re.I)


def sentences(text):
    """Предложения ПРОЗЫ. Строки внутри абзаца склеиваются (перенос строки не равен концу предложения),
    пункты списков, заголовки-ярлыки и служебные строки экспорта не считаются предложениями.
    Иначе средняя длина фразы занижается почти вдвое (проверено на корпусе: 8,6 против 11,6 слов)."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        prose = [l for l in lines if not LIST_LINE.match(l) and not SERVICE_LINE.match(l)]
        joined = " ".join(prose)
        if joined:
            out.extend(x.text.strip() for x in sentenize(joined) if x.text.strip())
    return out


def list_items(text):
    return [l.strip() for l in text.split("\n") if LIST_LINE.match(l)]


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def lemma(w):
    return MORPH.parse(w.lower())[0].normal_form


def pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


def per1k(n, total):
    return round(1000 * n / total, 2) if total else 0.0

# ---------- метрики ----------

FILLERS = ["ну", "вот", "короче", "как бы", "типа", "слушайте", "смотрите", "смотри",
           "в общем", "собственно", "кстати", "честно", "прям", "реально", "просто",
           "давайте", "знаете", "понимаете", "то есть", "значит", "так вот", "блин",
           "на самом деле", "наверное", "конечно", "да", "так сказать", "так или иначе", "какой-то", "какой то",
           "очень", "вообще", "именно", "сейчас", "уже"]
INTERJ = ["ох", "ах", "ух", "ого", "вау", "ммм", "хм", "ой", "эх", "ааа", "ну да", "ага", "угу", "фух", "упс"]
DISCOURSE_OPENERS = ["и", "а", "но", "так", "вот", "ну", "кстати", "кажется", "честно", "короче", "поэтому", "зато"]


def punct_profile(text, nwords):
    text = text.replace("\u2063", "")
    c = {
        "!": text.count("!"),
        "?": text.count("?"),
        "?!/!?": len(re.findall(r"[?!]{2,}", text)),
        "многоточие": len(re.findall(r"\.\.\.|…", text)),
        "многоточие после ? или !": len(re.findall(r"[?!]\.\.|[?!]…", text)),
        "двоеточие": text.count(":"),
        "точка с запятой": text.count(";"),
        "длинное тире —": text.count("—"),
        "короткое тире –": text.count("–"),
        "дефис как тире ( - )": len(re.findall(r"\s-\s", text)),
        "скобки (…)": len(re.findall(r"\([^)]{2,}\)", text)),
        "смайл-скобка )": len(re.findall(r"(?<![(\w])\)+(?!\w)", text)) - len(re.findall(r"\([^)]{2,}\)", text)),
        "кавычки «»": text.count("«"),
        "кавычки \"\"": text.count('"') // 2,
        "эмодзи": len(EMOJI_RE.findall(text)),
        "слова КАПСОМ": len(re.findall(r"\b[А-ЯЁ]{3,}\b", text)),
        "жирный **…**": len(re.findall(r"\*\*[^*]+\*\*", text)) ,
        "эмодзи-маркер в начале строки": len(re.findall(r"(?m)^\s*[\U0001F300-\U0001FAFF\u2600-\u27BF\u2705\u2714\u269C]", text)),
        "маркер списка • - 1.": len(re.findall(r"(?m)^\s*(•|-|\d+[.)])\s", text)),
        "слэш а/б (семья/дети)": len(re.findall(r"[а-яё]/[а-яё]", text, re.I)),
        "буква ё": len(re.findall(r"[ёЁ]", text)),
        "растянутые буквы (оооочень)": len(re.findall(r"([а-яё])\1{2,}", text, re.I)),
    }
    return {k: {"count": max(v, 0), "per_1k_words": per1k(max(v, 0), nwords)} for k, v in c.items()}


def openers(sents, docs_texts):
    first1 = Counter()
    first2 = Counter()
    for s in sents:
        w = words(s)
        if w:
            first1[w[0].lower()] += 1
        if len(w) >= 2:
            first2[" ".join(x.lower() for x in w[:2])] += 1
    post_open = Counter()
    for t in docs_texts:
        ss = sentences(t)
        if ss:
            w = words(ss[0])
            post_open[" ".join(x.lower() for x in w[:3])] += 1
    return first1, first2, post_open


def ngrams(sents, n, minc=3):
    c = Counter()
    for s in sents:
        w = [x.lower() for x in words(s)]
        for i in range(len(w) - n + 1):
            c[" ".join(w[i:i + n])] += 1
    return [(k, v) for k, v in c.most_common(60) if v >= minc]


def rhetoric(text_docs):
    res = Counter()
    examples = defaultdict(list)

    def ex(k, s):
        if len(examples[k]) < 6:
            examples[k].append(s[:220])

    for t in text_docs:
        ss = sentences(t)
        lines = [l.strip() for l in t.split("\n") if l.strip()]
        # анафора: 2+ подряд строк/предложений с одинаковым первым словом
        for seq in (ss, lines):
            for i in range(len(seq) - 1):
                a, b = words(seq[i]), words(seq[i + 1])
                if a and b and a[0].lower() == b[0].lower() and len(a[0]) > 1:
                    res["анафора"] += 1
                    ex("анафора", seq[i] + " / " + seq[i + 1])
        for i, s in enumerate(ss):
            low = s.lower()
            # вопрос-ответ: вопрос, за которым идёт короткий утвердительный ответ
            if s.endswith("?") and i + 1 < len(ss) and not ss[i + 1].endswith("?") and len(words(ss[i + 1])) <= 12:
                res["вопрос-ответ"] += 1
                ex("вопрос-ответ", s + " → " + ss[i + 1])
            # антитеза
            if re.search(r"\bне\b[^.!?]{1,60},\s*а\b", low) or re.search(r"\bне\b[^.!?]{1,40}\bа\b", low) \
                    or re.search(r",\s*а не\b", low) or re.search(r"\bвместо\b", low) or re.search(r"\bзато\b", low):
                res["антитеза"] += 1
                ex("антитеза", s)
            # градация / перечисление-нарастание: 3+ однородных через запятую
            if re.search(r"(\b[а-яё]+\b,\s*){2,}(и\s+)?\b[а-яё]+\b", low) and len(words(s)) <= 30:
                if re.search(r"\b(даже|ещё|еще|и наконец|а потом|больше|сильнее|быстрее)\b", low):
                    res["градация"] += 1
                    ex("градация", s)
                else:
                    res["перечисление-тройка"] += 1
                    ex("перечисление-тройка", s)
            # парцелляция: короткое предложение, начинающееся с союза
            w = words(s)
            if w and w[0].lower() in ("и", "а", "но", "или", "потому", "чтобы", "хотя") and len(w) <= 7 and i > 0:
                res["парцелляция"] += 1
                ex("парцелляция", ss[i - 1] + " | " + s)
            # риторическое обращение к читателю
            if re.search(r"\b(ты|вы|тебе|вам|тебя|вас)\b", low) and s.endswith("?"):
                res["вопрос к читателю"] += 1
                ex("вопрос к читателю", s)
            # однословные / номинативные предложения
            if len(w) <= 2 and w:
                res["предложение из 1–2 слов"] += 1
                ex("предложение из 1–2 слов", s)
    return res, examples


def syntax_profile(sents, limit=3000):
    """Порядок слов по зависимостям Natasha: подлежащее/сказуемое/дополнение."""
    try:
        from natasha import Segmenter, NewsEmbedding, NewsSyntaxParser, NewsMorphTagger, Doc
    except Exception:
        return None
    seg = Segmenter(); emb = NewsEmbedding()
    parser = NewsSyntaxParser(emb); tagger = NewsMorphTagger(emb)
    orders = Counter(); stats = Counter(); n = 0
    pos = Counter()
    for s in sents[:limit]:
        doc = Doc(s); doc.segment(seg); doc.parse_syntax(parser); doc.tag_morph(tagger)
        toks = doc.tokens
        if not toks:
            continue
        n += 1
        for t in toks:
            if t.pos and t.pos != "PUNCT":
                pos[t.pos] += 1
        root = next((t for t in toks if t.rel == "root"), None)
        if root is None:
            continue
        rid = root.id
        subj = next((t for t in toks if t.head_id == rid and t.rel in ("nsubj", "nsubj:pass")), None)
        obj = next((t for t in toks if t.head_id == rid and t.rel in ("obj", "iobj")), None)
        if root.pos not in ("VERB", "AUX"):
            stats["безглагольное (номинативное/оценочное)"] += 1
        if subj is None:
            stats["без подлежащего (безличное/неполное)"] += 1
        if subj is not None and root.pos == "VERB":
            idx = {t.id: i for i, t in enumerate(toks)}
            seq = sorted([("S", idx[subj.id]), ("V", idx[root.id])] + ([("O", idx[obj.id])] if obj else []), key=lambda x: x[1])
            orders["".join(x[0] for x in seq)] += 1
        if any(t.rel == "advcl" for t in toks): stats["с деепричастным/обстоятельственным оборотом"] += 1
        if any(t.rel == "acl:relcl" for t in toks): stats["с придаточным «который»"] += 1
        if any(t.rel == "parataxis" for t in toks): stats["с вводными/паратаксисом"] += 1
        if any(t.rel == "conj" and t.head_id == rid for t in toks): stats["с однородными сказуемыми"] += 1
    tot_pos = sum(pos.values())
    return {
        "sentences_parsed": n,
        "word_order": {k: pct(v, sum(orders.values())) for k, v in orders.most_common()},
        "constructions_pct": {k: pct(v, n) for k, v in stats.most_common()},
        "pos_pct": {k: pct(v, tot_pos) for k, v in pos.most_common(12)},
    }


LATIN_OK = re.compile(r"^[a-z\-]+$", re.I)


def typo_profile(docs_texts):
    unknown = Counter()
    for t in docs_texts:
        for tok in tokenize(t):
            w = tok.text
            if not CYR_RE.match(w) or len(w) < 3 or w[0].isupper():
                continue
            if not MORPH.word_is_known(w.lower()):
                unknown[w.lower()] += 1
    joined = "\n".join(docs_texts)
    patterns = {
        "строчная буква в начале предложения": len(re.findall(r"[.!?]\s+[а-яё]", joined)),
        "нет пробела после знака": len(re.findall(r"[а-яё][,.!?][а-яё]", joined, re.I)),
        "двойной пробел": joined.count("  "),
        "пробел перед знаком": len(re.findall(r"\s[,.!?:;]", joined)),
        "«что» без запятой перед ним": len(re.findall(r"[а-яё]\s+что\b", joined)) - len(re.findall(r"\b(не|потому|то|так|для того|вместо того|после того|при том|ни|чем|кроме того)\s+что\b", joined, re.I)),
        "«тся/ться» сомнительные": len(re.findall(r"\b(хочеться|кажеться|получаеться|нравиться мне)\b", joined, re.I)),
        "е вместо ё (все/всё и т.п.)": len(re.findall(r"\b(еще|ее|все-таки)\b", joined, re.I)),
    }
    return {"non_dictionary_words": unknown.most_common(80), "patterns": patterns}


def register_metrics(docs):
    texts = [re.sub(r"\((?:\.{1,3})\)", "", d["text"]) if d["register"] == "устный" else d["text"] for d in docs]
    texts = ["\n".join(l for l in t.splitlines() if not l.startswith("# ")) for t in texts]
    sents = [s for t in texts for s in sentences(t)]
    wlist = [w for s in sents for w in words(s)]
    nw = len(wlist) or 1
    slens = [len(words(s)) for s in sents if words(s)]
    plens = [len(words(p)) for t in texts for p in paragraphs(t)]
    dlens = [len(words(t)) for t in texts]
    low_all = " ".join(texts).lower()
    first_person = sum(1 for w in wlist if w.lower() in ("я", "мне", "меня", "мной", "мой", "моя", "моё", "мое", "мои"))
    we = sum(1 for w in wlist if w.lower() in ("мы", "нам", "нас", "наш", "наша", "наши"))
    you_t = sum(1 for w in wlist if w.lower() in ("ты", "тебе", "тебя", "твой", "твоя", "твои"))
    you_v = sum(1 for w in wlist if w.lower() in ("вы", "вам", "вас", "ваш", "ваша", "ваши"))
    lem = Counter(lemma(w) for w in wlist if len(w) > 3)
    f1, f2, popen = openers(sents, texts)
    rh, rh_ex = rhetoric(texts)
    return {
        "docs": len(texts), "sentences": len(sents), "words": nw,
        "sentence_len": {
            "mean": round(st.mean(slens), 1) if slens else 0,
            "median": st.median(slens) if slens else 0,
            "p90": sorted(slens)[int(len(slens) * 0.9)] if slens else 0,
            "share_<=5": pct(sum(1 for x in slens if x <= 5), len(slens)),
            "share_6-15": pct(sum(1 for x in slens if 6 <= x <= 15), len(slens)),
            "share_16-25": pct(sum(1 for x in slens if 16 <= x <= 25), len(slens)),
            "share_>25": pct(sum(1 for x in slens if x > 25), len(slens)),
        },
        "paragraph_len_words": {"mean": round(st.mean(plens), 1) if plens else 0, "median": st.median(plens) if plens else 0},
        "sentences_per_paragraph": round(len(sents) / max(len(plens), 1), 2),
        "doc_len_words": {"mean": round(st.mean(dlens), 1) if dlens else 0, "median": st.median(dlens) if dlens else 0},
        "sentence_types_pct": {
            "вопросительные": pct(sum(1 for s in sents if s.rstrip().endswith("?")), len(sents)),
            "восклицательные": pct(sum(1 for s in sents if s.rstrip().endswith("!")), len(sents)),
            "с многоточием в конце": pct(sum(1 for s in sents if re.search(r"(\.\.\.|…)$", s.rstrip())), len(sents)),
        },
        "pronouns_per_1k": {"я": per1k(first_person, nw), "мы": per1k(we, nw), "ты": per1k(you_t, nw), "вы": per1k(you_v, nw)},
        "lexical": {
            "type_token_ratio_lemmas": round(len(lem) / max(sum(lem.values()), 1), 3),
            "mean_word_len": round(st.mean(len(w) for w in wlist), 2) if wlist else 0,
            "top_lemmas": [(k, v) for k, v in lem.most_common(80)
                           if MORPH.parse(k)[0].tag.POS not in ("PREP", "CONJ", "PRCL", "NPRO", "INTJ")][:40],
        },
        "openers": {
            "sentence_first_word": f1.most_common(25),
            "sentence_first_2_words": f2.most_common(25),
            "post_first_3_words": popen.most_common(20),
            "discourse_opener_share_pct": pct(sum(f1[w] for w in DISCOURSE_OPENERS), len(sents)),
        },
        "ngrams": {"3": ngrams(sents, 3), "4": ngrams(sents, 4), "5": ngrams(sents, 5, 2)},
        "punctuation": punct_profile(" ".join(texts), nw),
        "fillers_per_1k": {f: per1k(len(re.findall(r"\b" + f + r"\b", low_all)), nw) for f in FILLERS},
        "interjections_per_1k": {f: per1k(len(re.findall(r"\b" + f + r"\b", low_all)), nw) for f in INTERJ},
        "rhetoric_per_1k_sentences": {k: per1k(v, len(sents)) for k, v in rh.most_common()},
        "rhetoric_examples": rh_ex,
        "_sents": sents, "_texts": texts,
        "ai_score_mean": round(st.mean(d.get("ai_score", 0) for d in docs), 2),
    }

# ---------- отчёт ----------

def table(d, h=("показатель", "значение")):
    rows = [f"| {h[0]} | {h[1]} |", "|---|---|"]
    for k, v in d.items():
        rows.append(f"| {k} | {v} |")
    return "\n".join(rows)


def compare_table(regs, getter, keys):
    names = list(regs)
    rows = ["| показатель | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k in keys:
        rows.append(f"| {k} | " + " | ".join(str(getter(regs[n], k)) for n in names) + " |")
    return "\n".join(rows)


def build_report(all_m, regs, syn, typos):
    L = []
    a = all_m
    L.append("# Корпусный профиль стиля\n")
    L.append(f"Корпус: {a['docs']} текстов, {a['sentences']} предложений, {a['words']} слов. "
             f"Регистры: " + ", ".join(f"{k} ({v['words']} сл.)" for k, v in regs.items()) + "\n")
    L.append("## 1. Длина фраз и ритм\n")
    L.append(table(a["sentence_len"]))
    L.append(f"\nАбзац: в среднем {a['paragraph_len_words']['mean']} слов, {a['sentences_per_paragraph']} предложения. "
             f"Текст: медиана {a['doc_len_words']['median']} слов.\n")
    L.append(table(a["sentence_types_pct"], ("тип предложения", "% от всех")))
    L.append("\n## 2. Зачины\n")
    L.append("Первое слово предложения: " + ", ".join(f"«{w}» {c}" for w, c in a["openers"]["sentence_first_word"]))
    L.append(f"\n\nДоля предложений, начатых дискурсивным словом (и/а/но/так/вот/ну/кстати…): **{a['openers']['discourse_opener_share_pct']}%**\n")
    L.append("Первые два слова: " + ", ".join(f"«{w}» {c}" for w, c in a["openers"]["sentence_first_2_words"]))
    L.append("\n\nКак начинаются тексты (первые 3 слова): " + ", ".join(f"«{w}» {c}" for w, c in a["openers"]["post_first_3_words"]))
    L.append("\n\n## 3. Повторяющиеся конструкции (n-граммы)\n")
    for n in ("5", "4", "3"):
        if a["ngrams"][n]:
            L.append(f"**{n}-словные:** " + "; ".join(f"«{k}» ×{v}" for k, v in a["ngrams"][n][:25]) + "\n")
    L.append("\n## 4. Синтаксис и порядок слов\n")
    if syn:
        L.append(f"Разобрано предложений: {syn['sentences_parsed']}\n")
        L.append("Порядок S (подлежащее) – V (сказуемое) – O (дополнение), % предложений с подлежащим:\n")
        L.append(table(syn["word_order"], ("порядок", "%")))
        L.append("\n" + table(syn["constructions_pct"], ("конструкция", "% предложений")))
        L.append("\nЧасти речи, % слов:\n")
        L.append(table(syn["pos_pct"], ("POS", "%")))
    L.append("\n## 5. Риторические фигуры (на 1000 предложений)\n")
    L.append(table(a["rhetoric_per_1k_sentences"], ("фигура", "на 1000 предл.")))
    for k, exs in a["rhetoric_examples"].items():
        L.append(f"\n**{k}** — примеры:\n" + "\n".join(f"- {e}" for e in exs))
    L.append("\n\n## 6. Пунктуация и графика (на 1000 слов)\n")
    L.append(table({k: v["per_1k_words"] for k, v in a["punctuation"].items()}, ("знак", "на 1000 слов")))
    L.append("\n## 7. Голос: местоимения, лексика\n")
    L.append(table(a["pronouns_per_1k"], ("местоимение", "на 1000 слов")))
    L.append(f"\nЛексическое разнообразие (TTR лемм): {a['lexical']['type_token_ratio_lemmas']}, средняя длина слова {a['lexical']['mean_word_len']}\n")
    L.append("Любимые слова (леммы): " + ", ".join(f"{k} {v}" for k, v in a["lexical"]["top_lemmas"]))
    L.append("\n\n## 8. Сравнение регистров\n")
    if len(regs) > 1:
        L.append(compare_table(regs, lambda r, k: r["sentence_len"][k], ["mean", "median", "share_<=5", "share_>25"]))
        L.append("\n" + compare_table(regs, lambda r, k: r["sentence_types_pct"][k], list(a["sentence_types_pct"])))
        L.append("\n" + compare_table(regs, lambda r, k: r["pronouns_per_1k"][k], ["я", "мы", "ты", "вы"]))
        L.append("\n" + compare_table(regs, lambda r, k: r["punctuation"][k]["per_1k_words"], list(a["punctuation"])))
        L.append("\n" + compare_table(regs, lambda r, k: r["rhetoric_per_1k_sentences"].get(k, 0), list(a["rhetoric_per_1k_sentences"])))
        L.append("\n" + compare_table(regs, lambda r, k: r["fillers_per_1k"][k], [f for f in FILLERS if a["fillers_per_1k"][f] > 0]))
        L.append("\n" + compare_table(regs, lambda r, k: r["paragraph_len_words"][k], ["mean", "median"]))
    else:
        L.append("Только один регистр в корпусе. Разложи тексты по подпапкам «продажи/», «устное/», чтобы получить сравнение.\n")
    L.append("\n## 9. Устное → письменное\n")
    L.append("Слова-паразиты и дискурсивные маркеры, на 1000 слов:\n")
    L.append(table({k: v for k, v in a["fillers_per_1k"].items() if v > 0}, ("маркер", "на 1000")))
    L.append("\nМеждометия:\n")
    L.append(table({k: v for k, v in a["interjections_per_1k"].items() if v > 0}, ("междометие", "на 1000")))
    L.append("\n## 10. Опечатки и ошибки\n")
    L.append(table(typos["patterns"], ("паттерн", "кол-во")))
    L.append("\nСлова вне словаря (кандидаты в привычные опечатки, сленг, неологизмы — нужна ручная проверка):\n")
    L.append(", ".join(f"{w} ×{c}" for w, c in typos["non_dictionary_words"]))
    return "\n".join(L)


def oral_block(docs):
    od = [d for d in docs if d["register"] == "устный"]
    if not od:
        return ""
    raw = "\n".join(d["text"] for d in od)
    raw = "\n".join(l for l in raw.splitlines() if not l.startswith("#"))
    segs = [s.strip() for s in re.split(r"\((?:\.{1,3})\)|\n", raw) if s.strip()]
    lens = [len(words(s)) for s in segs if words(s)]
    nw = sum(lens) or 1
    long_p = len(re.findall(r"\(\.\.\.?\)", raw)); short_p = len(re.findall(r"\(\.\)", raw))
    starts = Counter(words(s)[0].lower() for s in segs if words(s))
    tag_da = len(re.findall(r"\bда\b", raw))
    L = ["\n\n## 13. Устная речь: ритм по паузам\n",
         f"Расшифровано {nw} слов. Короткие паузы (0.3–0.7 c): {short_p}, длинные (>0.7 c): {long_p}.",
         f"Речевой отрезок между паузами: в среднем {round(st.mean(lens),1)} слов, медиана {st.median(lens)}.",
         f"Для сравнения, письменное предложение в среднем 8 слов: на письме речь режется в 2–3 раза мельче.\n",
         "С чего начинаются речевые отрезки: " + ", ".join(f"«{w}» {c}" for w, c in starts.most_common(20)),
         f"\nТег-вопрос / самоподтверждение «да» в речи: {per1k(tag_da, nw)} на 1000 слов.",
         "\nПервые отрезки (как звучит):\n"]
    L += [f"- {s}" for s in segs[:25]]
    return "\n".join(L)


def pre_gpt(d):
    """Пост до 30.11.2022 (выход ChatGPT): по гайду Википедии почти наверняка написан человеком."""
    m = re.match(r"(\d\d)\.(\d\d)\.(\d{4})", d.get("date", ""))
    return bool(m) and (int(m.group(3)), int(m.group(2)), int(m.group(1))) < (2022, 11, 30)


def own_voice_docs(docs):
    wd = [d for d in docs if d["register"] != "устный"]
    gold = [d for d in wd if pre_gpt(d)]
    srt = sorted(wd, key=lambda d: d["ai_score"])
    q = max(len(wd) // 3, min(5, len(wd)))
    low = srt[:q]
    if len(gold) >= 10:
        return gold, srt[-q:], f"эталон: {len(gold)} постов до 30.11.2022 (до ChatGPT)"
    base = gold + [d for d in low if d not in gold]
    return base, srt[-q:], f"эталон: {len(gold)} постов до ChatGPT + нижняя треть по индексу нейрослопа ({len(base)} всего)"


def own_voice_numbers(docs):
    base, _, note = own_voice_docs(docs)
    if not base:
        return {}
    texts = [d["text"] for d in base]
    ss = [x for t in texts for x in sentences(t)]
    sl = [len(words(x)) for x in ss if words(x)]
    nw = max(sum(len(words(t)) for t in texts), 1)
    j = " ".join(texts)
    return {
        "note": note,
        "слов в предложении": round(st.mean(sl), 1) if sl else 0,
        "жирный на 100 слов": round(100 * len(re.findall(r"\*\*[^*]+\*\*", j)) / nw, 2),
        "тире — на 100 слов": round(100 * len(re.findall(r"\s—\s", j)) / nw, 2),
        "эмодзи на 100 слов": round(100 * len(EMOJI_RE.findall(j)) / nw, 2),
        "индекс нейрослопа (медиана)": st.median([d["ai_score"] for d in base]),
    }


def ai_split_block(docs):
    wd = [d for d in docs if d["register"] != "устный"]
    if len(wd) < 12:
        return ""
    base, high, note = own_voice_docs(docs)
    cols = {"свой голос": quick(base), "скорее с ИИ (верхняя треть индекса)": quick(high)}
    L = ["\n\n## 14. Свой голос vs тексты, написанные с ИИ\n",
         f"Индекс нейрослопа считается по признакам гайда Википедии «Signs of AI writing» (slop_check.py). {note}.\n",
         compare_table(cols, lambda r, k: r[k], list(cols["свой голос"])),
         "\n\nСамые «свои» тексты (начало):\n"]
    wd = sorted(wd, key=lambda d: d["ai_score"])
    L += [f"- {d.get('date','')} · индекс {d['ai_score']} · {d['text'][:160].replace(chr(10),' / ')}" for d in wd[:8]]
    L.append("\nСамые «ИИ-шные» тексты (начало):\n")
    L += [f"- {d.get('date','')} · {d['ai_score']}/100 · {d['text'][:160].replace(chr(10),' / ')}" for d in wd[-6:]]
    return "\n".join(L)


def slop_block(docs):
    wd = [d for d in docs if d["register"] != "устный"]
    if not wd:
        return ""
    cat = Counter(); ids = Counter(); frags = defaultdict(Counter)
    nw = max(sum(len(d["text"].split()) for d in wd), 1)
    for d in wd:
        for h in slop_check.check(d["text"])["hits"]:
            ids[h["id"]] += 1; cat[h["cat"]] += 1; frags[h["id"]][h["frag"].lower()[:50]] += 1
    L = ["\n\n## 15. Нейрослоп в корпусе: какие признаки встречаются чаще всего\n",
         "| признак | на 1000 слов | частые примеры |", "|---|---|---|"]
    for k, v in ids.most_common(20):
        L.append(f"| {k} | {per1k(v, nw)} | " + "; ".join(f"«{f}»" for f, _ in frags[k].most_common(4)) + " |")
    years = defaultdict(list)
    for d in wd:
        if d.get("date"):
            years[d["date"][-4:]].append(d["ai_score"])
    if years:
        L.append("\nИндекс нейрослопа по годам (медиана): " + ", ".join(f"{y}: {st.median(v)}" for y, v in sorted(years.items())))
    return "\n".join(L)


def quick(docs):
    texts = [d["text"] for d in docs]
    ss = [x for t in texts for x in sentences(t)]
    wl = [w for x in ss for w in words(x)]
    nw = len(wl) or 1
    sl = [len(words(x)) for x in ss if words(x)]
    j = " ".join(texts)
    return {
        "текстов": len(texts),
        "слов в тексте (медиана)": st.median([len(words(t)) for t in texts]) if texts else 0,
        "слов в предложении": round(st.mean(sl), 1) if sl else 0,
        "? на 1k": per1k(j.count("?"), nw), "! на 1k": per1k(j.count("!"), nw),
        "эмодзи на 1k": per1k(len(EMOJI_RE.findall(j)), nw),
        "жирный на 1k": per1k(len(re.findall(r"\*\*[^*]+\*\*", j)), nw),
        "тире — на 1k": per1k(j.count("—") + len(re.findall(r"\s-\s", j)), nw),
        "многоточие на 1k": per1k(len(re.findall(r"\.\.\.|…", j)), nw),
        "«я» на 1k": per1k(sum(w.lower() in ("я", "мне", "меня", "мой", "моя", "мои") for w in wl), nw),
        "«вы» на 1k": per1k(sum(w.lower() in ("вы", "вам", "вас", "ваш", "ваша", "ваши") for w in wl), nw),
        "«ты» на 1k": per1k(sum(w.lower() in ("ты", "тебе", "тебя", "твой", "твоя", "твои") for w in wl), nw),
    }


def evolution_and_reactions(docs):
    L = []
    dated = [d for d in docs if d.get("date") and d["register"] != "устный"]
    if dated:
        years = sorted(set(d["date"][-4:] for d in dated))
        cols = {y: quick([d for d in dated if d["date"][-4:] == y]) for y in years}
        L.append("\n\n## 11. Эволюция стиля по годам\n")
        L.append(compare_table(cols, lambda r, k: r[k], list(next(iter(cols.values())))))
    rd = [d for d in docs if isinstance(d.get("reactions"), int) and d["register"] != "устный"]
    if len(rd) >= 20:
        rd.sort(key=lambda d: d["reactions"], reverse=True)
        q = max(len(rd) // 4, 5)
        cols = {"топ-25% по реакциям": quick(rd[:q]), "нижние 25%": quick(rd[-q:])}
        L.append("\n\n## 12. Что отличает тексты, которые собирают больше реакций\n")
        L.append(compare_table(cols, lambda r, k: r[k], list(cols["нижние 25%"])))
        L.append("\n\nТоп-5 постов по реакциям (первая строка):\n")
        for d in rd[:5]:
            L.append(f"- {d['date']} · {d['reactions']} реакций · {d['text'].splitlines()[0][:120]}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-syntax", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "_анализ_стиля")
    os.makedirs(out, exist_ok=True)
    docs = load_corpus(a.root)
    docs = [d for d in docs if "_анализ_стиля" not in d.get("src", "")]
    if not docs:
        sys.exit("Текстов не найдено")
    all_m = register_metrics(docs)
    regs = {}
    for r in sorted(set(d["register"] for d in docs)):
        regs[r] = register_metrics([d for d in docs if d["register"] == r])
    written = [s for r, m in regs.items() if r != "устный" for s in m["_sents"]] or all_m["_sents"]
    syn = None if a.no_syntax else syntax_profile(written)
    typos = typo_profile([d["text"] for d in docs if d["register"] != "устный"] or all_m["_texts"])
    report = build_report(all_m, regs, syn, typos)
    report += evolution_and_reactions(docs) + oral_block(docs) + ai_split_block(docs) + slop_block(docs)
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(report)

    def clean(m):
        return {k: v for k, v in m.items() if not k.startswith("_")}
    prof = {"all": clean(all_m), "registers": {k: clean(v) for k, v in regs.items()}, "syntax": syn, "typos": typos,
            "own_voice": own_voice_numbers(docs)}
    json.dump(prof, open(os.path.join(out, "profile.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"OK: {len(docs)} текстов → {out}/report.md, profile.json")


if __name__ == "__main__":
    main()
