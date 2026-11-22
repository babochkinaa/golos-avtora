#!/usr/bin/env python3
"""
Выгрузка Telegram → корпус для style_analyzer.py

Понимает: экспорт Telegram Desktop в HTML (messages.html, messages2.html, …),
в JSON (result.json), zip-архив с экспортом, папку с экспортом.

    python3 tg_to_corpus.py <файл|папка|zip> [ещё…] --out corpus [--author "Имя"]

--author нужен для групп и чатов: оставит только сообщения этого человека.
Для канала не нужен: если авторов несколько, берётся самый частый (с предупреждением).

На выходе: corpus/posts.jsonl (текст, дата, реакции, медиа, регистр),
corpus/повседневное/posts.txt, corpus/продажи/posts.txt
"""
import sys, os, re, json, argparse, zipfile, tempfile, glob
from collections import Counter

SALES = re.compile(r"(старт|стартуе|мест[а]? (осталось|мало)|осталось \d+ мест|запис[ыа]|оплат|стоимост|цена|₽|руб\b|тариф|"
                   r"предзапис|по ссылке|пиши(те)? (мне )?(в личк|в лс|в директ|слово)|кодовое слово|бронь|забронир|"
                   r"практикум|интенсив|курс[еау]? |марафон|наставничеств|консультаци|"
                   r"до \d+ (ноября|декабря|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября)|"
                   r"скидк|успей|последний день|набор|поток|оффер|купить|заказ)", re.I)


def classify(txt):
    return "продающий" if len(SALES.findall(txt)) >= 2 else "повседневный"


def collect(inputs):
    files = []
    tmp = None
    for inp in inputs:
        if inp.lower().endswith(".zip"):
            tmp = tmp or tempfile.mkdtemp()
            with zipfile.ZipFile(inp) as z:
                for n in z.namelist():
                    if n.lower().endswith((".html", ".json")) and "/." not in n:
                        z.extract(n, tmp)
            inp = tmp
        if os.path.isdir(inp):
            files += glob.glob(os.path.join(inp, "**", "messages*.html"), recursive=True)
            files += glob.glob(os.path.join(inp, "**", "result.json"), recursive=True)
        else:
            files.append(inp)

    def order(f):
        m = re.search(r"messages(\d*)\.html", f)
        return (os.path.dirname(f), int(m.group(1) or 1) if m else 0)
    return sorted(set(files), key=order)


def from_json(path):
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue
        parts = m.get("text", "")
        if isinstance(parts, str):
            txt = parts
        else:
            buf = []
            for p in parts:
                if isinstance(p, str):
                    buf.append(p)
                else:
                    t = p.get("text", "")
                    buf.append(f"**{t}**" if p.get("type") == "bold" else f"_{t}_" if p.get("type") == "italic" else t)
            txt = "".join(buf)
        reacts = sum(r.get("count", 0) for r in m.get("reactions", []) or [])
        d = m.get("date", "")[:10]
        date = f"{d[8:10]}.{d[5:7]}.{d[:4]}" if len(d) == 10 else ""
        media = [k for k in ("photo", "media_type", "file") if m.get(k)]
        out.append({"date": date, "text": txt.strip(), "reactions": reacts, "media": media,
                    "author": m.get("from", "") or ""})
    return out


def from_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "html.parser")
    out, last_author = [], ""
    for m in soup.select("div.message.default"):
        fn = m.select_one("div.from_name")
        if fn:
            last_author = re.sub(r"\s+", " ", fn.get_text()).strip()
        t = m.select_one("div.text")
        if not t:
            continue
        for br in t.find_all("br"):
            br.replace_with("\n")
        for b in t.find_all(["strong", "b"]):
            b.replace_with("**" + b.get_text() + "**")
        for i in t.find_all(["em", "i"]):
            i.replace_with("_" + i.get_text() + "_")
        txt = re.sub(r"[ \t]+\n", "\n", t.get_text()).strip()
        d = m.select_one("div.date")
        reacts = 0
        for c in m.select("span.count"):
            reacts += int(re.sub(r"\D", "", c.get_text()) or 0)
        media = [x for x in ("photo", "video", "voice", "round", "file") if m.select_one(f".media_{x}")]
        out.append({"date": d["title"][:10] if d and d.has_attr("title") else "", "text": txt,
                    "reactions": reacts, "media": media, "author": last_author})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--author", default=None)
    ap.add_argument("--min-words", type=int, default=5)
    a = ap.parse_args()
    files = collect(a.inputs)
    if not files:
        sys.exit("Не нашла messages*.html или result.json")
    posts = []
    for f in files:
        posts += from_json(f) if f.endswith(".json") else from_html(f)
    authors = Counter(p["author"] for p in posts if p["author"])
    if a.author:
        posts = [p for p in posts if a.author.lower() in p["author"].lower()]
    elif len(authors) > 1:
        top, n = authors.most_common(1)[0]
        print(f"⚠ Авторов несколько: {dict(authors.most_common(5))}. Беру самого частого: «{top}». "
              f"Если это не тот человек, перезапусти с --author.", file=sys.stderr)
        posts = [p for p in posts if p["author"] == top]
    posts = [p for p in posts if len(p["text"].split()) >= a.min_words]
    seen, uniq = set(), []
    for p in posts:
        k = p["text"][:200]
        if k not in seen:
            seen.add(k); uniq.append(p)
    posts = uniq
    for p in posts:
        p["register"] = classify(p["text"])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "posts.jsonl"), "w", encoding="utf-8") as fh:
        for p in posts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    for reg, folder in (("повседневный", "повседневное"), ("продающий", "продажи")):
        os.makedirs(os.path.join(a.out, folder), exist_ok=True)
        open(os.path.join(a.out, folder, "posts.txt"), "w", encoding="utf-8").write(
            "\n\n-----\n\n".join(p["text"] for p in posts if p["register"] == reg))
    nw = sum(len(p["text"].split()) for p in posts)
    ns = sum(p["register"] == "продающий" for p in posts)
    years = Counter(p["date"][-4:] for p in posts if p["date"])
    print(f"Файлов: {len(files)}. Постов: {len(posts)} ({len(posts) - ns} повседневных, {ns} продающих), слов: {nw}. "
          f"По годам: {dict(sorted(years.items()))}")
    if nw < 5000:
        print("⚠ Меньше 5000 слов: цифры будут неустойчивыми. Лучше 15–20 тыс. слов и больше.", file=sys.stderr)


if __name__ == "__main__":
    main()
